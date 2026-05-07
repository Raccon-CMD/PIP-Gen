import os
import torch
import configparser
import pandas as pd
import torch.nn.functional as F
from UnetDiff.models.UNetDenoiser import UNetDenoiser
from UnetDiff.models.layers.EMA import EMA
from UnetDiff.utils.DiffDataset import XYDataset
from UnetDiff.utils.utils import set_seed, extract
from transformers import EsmTokenizer, EsmModel
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch import device
from torch.cuda import is_available

# ---------- Globals ----------
device = device("cuda:0" if is_available() else "cpu")

# ---------- Config reading ----------
def read_config():
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        r'D:\python code\PIP-Gen\config.ini',
        '../config.ini',
        os.path.join(script_dir, '../config.ini'),
        os.path.join(script_dir, '../../config.ini'),
    ]
    config_path = None
    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            config_path = abs_path
            break
    if config_path is None:
        raise FileNotFoundError("config.ini not found!")

    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(config_path, encoding='utf-8')
    return dict(conf.items('UnetDiff_conf'))

# ---------- Main training logic ----------
def main():
    conf_dict = read_config()
    set_seed(int(conf_dict['SEED']))

    # Noise schedule
    timesteps = int(conf_dict['TIME_STEPS'])
    t = torch.arange(1, timesteps + 1, dtype=torch.long, device=device)
    alphas_cumprod = 1 - torch.sqrt(t / (timesteps + float(conf_dict['SQRT_S'])))
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # Model & Data
    # 使用绝对路径，避免相对路径中的 '..' 引起 repo_id 校验错误
    LOCAL_ESM = r'D:\python code\PIP-Gen\prediction\ESM2-8M'
    tokenizer = EsmTokenizer.from_pretrained(LOCAL_ESM, trust_remote_code=True)
    esm2_model = EsmModel.from_pretrained(LOCAL_ESM, trust_remote_code=True, add_pooling_layer=True).to(device)
    esm2_model.eval()

    train_dataset = XYDataset(pd.read_csv(r'D:\python code\PIP-Gen\data\Generation Data\Gen_proinflammatory_Train.csv'))
    val_dataset = XYDataset(pd.read_csv(r'D:\python code\PIP-Gen\data\Generation Data\Gen_proinflammatory-Val.csv'))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(conf_dict['DIFFUSION_BATCH_SIZE']),
        shuffle=True, pin_memory=True, num_workers=8, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1024, pin_memory=True, num_workers=8, persistent_workers=True)

    # Denoiser
    denoiser_mlp = [int(i) for i in conf_dict['DENOISER_MLP'].split(',')]
    denoiser = UNetDenoiser('../' + conf_dict['DENOISER_ESM_MODEL_NAME'],
                        int(conf_dict['DENOISER_EMBEDDING']), denoiser_mlp).to(device)
    optimizer = torch.optim.AdamW(denoiser.parameters(),
                                  lr=float(conf_dict['DENOISER_LR']),
                                  weight_decay=float(conf_dict['DENOISER_WEIGHT_DECAY']))
    scheduler = CosineLRScheduler(optimizer, t_initial=200_000,
                                  lr_min=float(conf_dict['MIN_DENOISER_LR']),
                                  warmup_lr_init=1e-8, warmup_t=10_000,
                                  cycle_limit=1, t_in_epochs=False)
    ema = EMA(denoiser, 0.99)
    ema.register()

    os.makedirs('../save_model', exist_ok=True)
    writer = SummaryWriter(log_dir='./logs')

    # Resume from checkpoint
    start_epoch, x_steps = 0, 1
    if os.path.exists("../save_model/checkpoint.pth"):
        ckpt = torch.load("../save_model/checkpoint.pth")
        denoiser.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        ema = ckpt['ema']
        start_epoch = ckpt['epoch'] + 1
        x_steps = ckpt['x_steps']

    # Training loop
    pbar = tqdm(total=len(train_loader))
    for epoch in range(start_epoch, int(conf_dict['DENOISER_EPOCH'])):
        if epoch % 100 == 0:
            set_seed(int(conf_dict['SEED']) + epoch)
        if epoch % 20 == 0:
            writer = SummaryWriter(log_dir='./logs')
        denoiser.train()
        for idx, datas in enumerate(train_loader):
            seq_enc = tokenizer(datas['sequences'],
                                max_length=int(conf_dict['MAX_LENGTH']) + 2,
                                padding='max_length',
                                return_tensors="pt")
            seq_ids, mask = seq_enc['input_ids'].to(device), seq_enc['attention_mask'].to(device)
            with torch.no_grad():
                x0 = esm2_model(seq_ids, mask).last_hidden_state

            t = torch.randint(1, int(conf_dict['TIME_STEPS']) + 1,
                              (x0.shape[0],), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            sqrt_a = extract(sqrt_alphas_cumprod, t - 1, x0.shape)
            sqrt_ma = extract(sqrt_one_minus_alphas_cumprod, t - 1, x0.shape)
            x_t = sqrt_a * x0 + sqrt_ma * noise

            optimizer.zero_grad()
            pred_x0 = denoiser(x_t, t, y=datas['labels'].to(device), attention_mask=mask)
            loss = F.mse_loss(pred_x0, x0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(),
                                           max_norm=float(conf_dict['DENOISER_CLIP_GRAD']))
            optimizer.step()
            ema.update()

            if x_steps % 20 == 0:
                pbar.set_description(f"epoch:{epoch + 1}  loss:{loss.item():.4f}")
                writer.add_scalar("train/x0_loss", loss.item(), x_steps)
            x_steps += 1
            scheduler.step_update(x_steps)
            pbar.update()
        pbar.reset()

        # Validation
        ema.apply_shadow()
        denoiser.eval()
        val_losses = []
        with torch.no_grad():
            for datas in val_loader:
                seq_enc = tokenizer(datas['sequences'],
                                    max_length=int(conf_dict['MAX_LENGTH']) + 2,
                                    padding='max_length',
                                    return_tensors="pt")
                seq_ids, mask = seq_enc['input_ids'].to(device), seq_enc['attention_mask'].to(device)
                x0 = esm2_model(seq_ids, mask).last_hidden_state
                t = torch.randint(0, int(conf_dict['TIME_STEPS']),
                                  (x0.shape[0],), device=device).long()
                noise = torch.randn_like(x0)
                sqrt_a = extract(sqrt_alphas_cumprod, t, x0.shape)
                sqrt_ma = extract(sqrt_one_minus_alphas_cumprod, t, x0.shape)
                x_t = sqrt_a * x0 + sqrt_ma * noise
                pred_x0 = denoiser(x_t, t, y=datas['labels'].to(device),
                                   attention_mask=mask)
                val_losses.append(F.mse_loss(pred_x0, x0).item())
        val_loss = torch.tensor(val_losses).mean()
        writer.add_scalar("val/x0_loss", val_loss.item(), epoch + 1)

        # Save models
        if (epoch + 1) % 20 == 0:
            torch.save(denoiser.state_dict(),
                       "../save_model/denoiser_model.pkl")
        torch.save({'model_state_dict': denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'ema': ema,
                    'epoch': epoch,
                    'x_steps': x_steps},
                   "../save_model/checkpoint.pth")
        epoch += 1
        ema.restore()

    # Training finished
    ema.apply_shadow()
    torch.save(denoiser.state_dict(),
               "../save_model/denoiser_model.pkl")
    print("Proinflammatory peptide denoiser training completed! Saved to: ../save_model/denoiser_model.pkl")

# ---------- Windows multiprocess guard ----------
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    main()