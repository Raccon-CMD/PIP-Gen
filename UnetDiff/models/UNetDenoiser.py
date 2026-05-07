import torch
from torch import nn
import torch.nn.functional as F
from .layers.LabelEmbedder import LabelEmbedder
from .layers.SinusoidalPositionEmbeddings import SinusoidalPositionEmbeddings


class UNetDenoiser(nn.Module):
    def __init__(self, esm_model_path, denoiser_embedding, denoiser_mlp):
        super(UNetDenoiser, self).__init__()
        self.denoiser_embedding = denoiser_embedding

        # 时间嵌入
        self.time_emb = nn.Sequential(
            SinusoidalPositionEmbeddings(denoiser_embedding),
            nn.Linear(denoiser_embedding, denoiser_embedding * 4),
            nn.SiLU(),
            nn.Linear(denoiser_embedding * 4, denoiser_embedding * 4),
        )

        # 标签嵌入
        self.label_emb = LabelEmbedder(3, denoiser_embedding * 4, 0.2)

        # 简化的U-Net架构
        # 编码器：3个残差块 + 2个下采样
        self.encoders = nn.ModuleList([
            ResidualBlock(128, 128, denoiser_embedding * 4),
            ResidualBlock(128, 128, denoiser_embedding * 4),
            ResidualBlock(128, 128, denoiser_embedding * 4),
        ])

        # 下采样层
        self.downsamplers = nn.ModuleList([
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
        ])

        # 中间层
        self.middle = ResidualBlock(128, 128, denoiser_embedding * 4)

        # 解码器：3个残差块 + 2个上采样
        self.decoders = nn.ModuleList([
            ResidualBlock(256, 128, denoiser_embedding * 4),  # 第一个解码器：256→128
            ResidualBlock(256, 128, denoiser_embedding * 4),  # 第二个解码器：256→128
            ResidualBlock(256, 128, denoiser_embedding * 4),  # 第三个解码器：256→128
        ])

        # 上采样层：只有2个，对应前两个解码器块
        self.upsamplers = nn.ModuleList([
            UpsampleLayer(128, 128),  # 第一个上采样
            UpsampleLayer(128, 128),  # 第二个上采样
        ])

        # 输入输出投影层
        self.input_proj = nn.Conv1d(denoiser_embedding, 128, kernel_size=1)
        self.output_proj = nn.Conv1d(128, denoiser_embedding, kernel_size=1)

        # 条件投影层：7个（3编码器 + 1中间 + 3解码器）
        self.time_proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(denoiser_embedding * 4, 128 * 2)  # 输出scale和shift
            ) for _ in range(7)
        ])

    def forward(self, x, time, y, attention_mask=None, return_attn_matrix=False):
        # 1. 输入形状调整 [B, L, D] → [B, D, L]
        x = x.transpose(1, 2)

        # 2. 条件嵌入（时间+标签）
        time_emb = self.time_emb(time)
        label_emb = self.label_emb(y, self.training)
        c = time_emb + label_emb  # 融合条件

        # 3. 输入投影
        x = self.input_proj(x)

        # 4. 编码器路径（下采样）
        skips = []  # 保存跳跃连接
        time_proj_idx = 0

        # 第一个编码器块（无下采样）
        scale_shift = self.time_proj_layers[time_proj_idx](c)
        time_proj_idx += 1
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.encoders[0](
            x,
            scale.unsqueeze(-1),  # [B, 128, 1]
            shift.unsqueeze(-1)  # [B, 128, 1]
        )
        skips.append(x)  # 保存跳跃连接1

        # 后续编码器块（带下采样）
        for i in range(1, len(self.encoders)):
            # 下采样
            x = self.downsamplers[i - 1](x)  # 序列长度减半

            # 条件投影
            scale_shift = self.time_proj_layers[time_proj_idx](c)
            time_proj_idx += 1
            scale, shift = scale_shift.chunk(2, dim=1)

            # 编码器块
            x = self.encoders[i](
                x,
                scale.unsqueeze(-1),
                shift.unsqueeze(-1)
            )
            skips.append(x)  # 保存跳跃连接

        # 5. 中间层（瓶颈）
        scale_shift = self.time_proj_layers[time_proj_idx](c)
        time_proj_idx += 1
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.middle(x, scale.unsqueeze(-1), shift.unsqueeze(-1))

        # 6. 解码器路径（上采样 + 跳跃连接）
        # 注意：有3个解码器块，但只有2个上采样器
        # 前两个解码器块：先上采样，再与跳跃连接拼接
        # 最后一个解码器块：直接与跳跃连接拼接（无上采样）

        # 解码器块1（带第一个上采样）
        x = self.upsamplers[0](x)  # 上采样1：L/4 → L/2
        skip = skips[-1]  # 取最后一个跳跃连接（编码器块3的输出）

        # 尺寸对齐
        if x.size(2) != skip.size(2):
            skip = F.interpolate(skip, size=x.size(2), mode='linear', align_corners=False)

        x = torch.cat([x, skip], dim=1)  # 拼接：[128, L/2] + [128, L/2] = [256, L/2]

        # 条件注入
        scale_shift = self.time_proj_layers[time_proj_idx](c)
        time_proj_idx += 1
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.decoders[0](x, scale.unsqueeze(-1), shift.unsqueeze(-1))

        # 解码器块2（带第二个上采样）
        x = self.upsamplers[1](x)  # 上采样2：L/2 → L
        skip = skips[-2]  # 取编码器块2的输出

        # 尺寸对齐
        if x.size(2) != skip.size(2):
            skip = F.interpolate(skip, size=x.size(2), mode='linear', align_corners=False)

        x = torch.cat([x, skip], dim=1)  # 拼接

        # 条件注入
        scale_shift = self.time_proj_layers[time_proj_idx](c)
        time_proj_idx += 1
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.decoders[1](x, scale.unsqueeze(-1), shift.unsqueeze(-1))

        # 解码器块3（无上采样，直接拼接最后一个跳跃连接）
        skip = skips[-3]  # 取编码器块1的输出

        # 尺寸对齐（通常已匹配，因为序列长度已恢复为L）
        if x.size(2) != skip.size(2):
            skip = F.interpolate(skip, size=x.size(2), mode='linear', align_corners=False)

        x = torch.cat([x, skip], dim=1)  # 拼接

        # 条件注入
        scale_shift = self.time_proj_layers[time_proj_idx](c)
        time_proj_idx += 1
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.decoders[2](x, scale.unsqueeze(-1), shift.unsqueeze(-1))

        # 7. 输出投影
        x = self.output_proj(x)

        # 8. 恢复原始形状 [B, D, L] → [B, L, D]
        x = x.transpose(1, 2)

        if return_attn_matrix:
            return x, []
        else:
            return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 第一个卷积块：GroupNorm + SiLU + Conv1d
        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        # 第二个卷积块
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # 残差连接
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

        # 注意：这里定义了time_emb_proj，但在当前设计中不被使用
        # 保留它但不调用，或者可以删除
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )

    def forward(self, x, scale=None, shift=None):
        residual = self.residual(x)

        # 第一个卷积
        x = self.conv1(x)

        # 应用条件缩放和偏移（从外部传入）
        if scale is not None and shift is not None:
            # scale + 1.0 确保初始缩放因子为1
            x = x * (scale + 1.0) + shift

        # 第二个卷积
        x = self.conv2(x)

        # 残差连接
        return x + residual


class UpsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 线性插值上采样
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        x = self.conv(x)
        return x