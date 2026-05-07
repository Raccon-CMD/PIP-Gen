from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
import numpy as np
from torch.utils.data import random_split
from LossFunction.focalLoss import FocalLoss_v2
import random


# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Use fully connected layers instead of 1x1 convolution
        self.fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [B, L, C]
        b, l, c = x.size()

        # Average pooling
        avg_out = self.avg_pool(x.permute(0, 2, 1))  # [B, C, 1]
        avg_out = avg_out.view(b, c)  # [B, C]
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))  # [B, C]

        # Max pooling
        max_out = self.max_pool(x.permute(0, 2, 1))  # [B, C, 1]
        max_out = max_out.view(b, c)  # [B, C]
        max_out = self.fc2(self.relu(self.fc1(max_out)))  # [B, C]

        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)  # [B, C]
        out = out.unsqueeze(1)  # [B, 1, C]

        # Apply attention weights
        return x * out.expand_as(x)


# Collate function for padding
def coll_paddding(batch_traindata):
    batch_traindata.sort(key=lambda data: len(data[0]), reverse=True)
    feature0 = []
    f0agv = []
    feature_fusion = []
    train_y = []
    for data in batch_traindata:
        feature0.append(data[0])
        f0agv.append(data[1])
        feature_fusion.append(data[2])
        train_y.append(data[3])
    data_length = [len(data) for data in feature0]
    mask = torch.full((len(batch_traindata), data_length[0]), False).bool()
    for mi, aci in zip(mask, data_length):
        mi[aci:] = True
    feature0 = torch.nn.utils.rnn.pad_sequence(feature0, batch_first=True, padding_value=0)
    f0agv = torch.nn.utils.rnn.pad_sequence(f0agv, batch_first=True, padding_value=0)
    feature_fusion = torch.nn.utils.rnn.pad_sequence(feature_fusion, batch_first=True, padding_value=0)
    train_y = torch.nn.utils.rnn.pad_sequence(train_y, batch_first=True, padding_value=0)
    return feature0, f0agv, feature_fusion, train_y, torch.tensor(data_length)


# Dataset class
class BioinformaticsDataset(Dataset):
    def __init__(self, X_prot, X_feature_fusion, mode='train'):
        self.X_prot = X_prot
        self.X_feature_fusion = X_feature_fusion
        self.mode = mode  # 'train' or 'test'

    def __getitem__(self, index):
        filename_prot = self.X_prot[index]
        df_prot = pd.read_csv(filename_prot)
        prot = df_prot.iloc[:, 1:].values
        if prot.dtype == object:
            prot = prot.astype(float)
        prot = torch.tensor(prot, dtype=torch.float)
        agv = torch.mean(prot, dim=0)
        agv = agv.repeat(prot.shape[0], 1)

        filename_feature_fusion = self.X_feature_fusion[index]
        df_feature_fusion = pd.read_csv(filename_feature_fusion)
        feature_fusion = df_feature_fusion.iloc[:, 1:].values
        feature_fusion = torch.tensor(feature_fusion, dtype=torch.float)

        label = df_prot.iloc[:, 0].values
        label = torch.tensor(label, dtype=torch.long)

        # Data augmentation for training
        if self.mode == 'train':
            # Random masking augmentation
            if torch.rand(1) < 0.3:  # 30% probability
                mask_len = max(1, int(0.15 * prot.shape[0]))
                start_idx = torch.randint(0, max(1, prot.shape[0] - mask_len), (1,))
                prot[start_idx:start_idx + mask_len] = 0

            # Feature jittering augmentation
            if torch.rand(1) < 0.2:  # 20% probability
                noise = torch.randn_like(prot) * 0.05
                prot += noise

        return prot, agv, feature_fusion, label

    def __len__(self):
        return len(self.X_prot)


class PIPModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(True)

        # 2. Channel attention (replaces self-attention)
        self.channel_attn_prot = ChannelAttention(in_channels=480, reduction_ratio=8)
        self.channel_attn_evo = ChannelAttention(in_channels=480, reduction_ratio=8)

        # 3. Bidirectional GRU
        self.bi_gru_prot = nn.GRU(480, 480, bidirectional=True, batch_first=True)
        self.bi_gru_evo = nn.GRU(480, 480, bidirectional=True, batch_first=True)

        # 4. Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(1920, 480),
            nn.ReLU(),
            nn.LayerNorm(480),
            nn.Dropout(0.3)
        )

        # 5. Convolutional backbone
        self.protcnn1 = nn.Conv1d(480 + 480 + 480, 768, 3, padding='same')
        self.bn1 = nn.BatchNorm1d(768)
        self.protcnn2 = nn.Conv1d(768, 384, 3, padding='same')
        self.bn2 = nn.BatchNorm1d(384)
        self.protcnn3 = nn.Conv1d(384, 192, 3, padding='same')
        self.bn3 = nn.BatchNorm1d(192)

        # 6. Main output
        self.fc2 = nn.Linear(192, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 2)
        self.drop = nn.Dropout(0.5)

        # 7. Specificity optimization branch
        self.specificity_head = nn.Sequential(
            nn.Conv1d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [B, 64, 1]
            nn.Flatten(),  # [B, 64]
            nn.Linear(64, 2)  # [B, 2]
        )

    def forward(self, prot0, f0agv, evo, data_length):
        # ---------- Channel attention & Bi-GRU ----------
        # Apply channel attention
        prot_attn = self.channel_attn_prot(prot0)
        evo_attn = self.channel_attn_evo(evo)

        prot_packed = nn.utils.rnn.pack_padded_sequence(
            prot_attn, data_length.cpu(), batch_first=True, enforce_sorted=False)
        evo_packed = nn.utils.rnn.pack_padded_sequence(
            evo_attn, data_length.cpu(), batch_first=True, enforce_sorted=False)

        prot_gru, _ = self.bi_gru_prot(prot_packed)
        evo_gru, _ = self.bi_gru_evo(evo_packed)

        prot_gru, _ = nn.utils.rnn.pad_packed_sequence(prot_gru, batch_first=True)
        evo_gru, _ = nn.utils.rnn.pad_packed_sequence(evo_gru, batch_first=True)

        fused = torch.cat([prot_gru, evo_gru], dim=-1)
        fused = self.fusion(fused)

        # ---------- Remove context attention, directly concatenate features ----------
        cat_feat = torch.cat((prot0, f0agv, fused), dim=2)

        # ---------- Convolutional backbone ----------
        conv_out = cat_feat.permute(0, 2, 1)
        conv_out = self.relu(self.bn1(self.protcnn1(conv_out)))
        conv_out = self.relu(self.bn2(self.protcnn2(conv_out)))
        conv_out = self.relu(self.bn3(self.protcnn3(conv_out)))  # [B, 192, L]

        # ---------- Main output ----------
        x = conv_out.permute(0, 2, 1)
        x = self.relu(self.drop(self.fc2(x)))
        x = self.relu(self.drop(self.fc3(x)))
        x_main = self.fc4(x)  # [B, L, 2]

        # ---------- Specificity branch ----------
        x_spec = self.specificity_head(conv_out)  # [B, 2]

        # ---------- Fused output ----------
        # Expand x_spec to [B, L, 2] and fuse
        x_spec = x_spec.unsqueeze(1).expand_as(x_main)
        return 0.7 * x_main + 0.3 * x_spec


# Training function
def train():
    # 1) Build dataset and split
    full_dataset = BioinformaticsDataset(prot_train + prot_test,
                                         fusion_train + fusion_test,
                                         mode='train')
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Training set size: {len(train_set)}, Validation set size: {len(val_set)}")

    # 2) Data loaders
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=coll_paddding
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=coll_paddding
    )

    # 3) Model, optimizer, loss, scheduler
    model = PIPModule().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.06)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    per_cls_weights = torch.FloatTensor([0.25, 0.75]).to(device)
    fcloss = FocalLoss_v2(alpha=per_cls_weights, gamma=2)

    # 4) Training loop
    best_val_loss, counter, patience = float('inf'), 0, 30
    epochs = 80
    for epoch in range(epochs):
        # ==================== Training phase ====================
        model.train()
        epoch_train_loss, train_correct, train_total, nb_train = 0.0, 0, 0, 0
        for prot_x, f0agv, evo_x, data_y, length in train_loader:
            optimizer.zero_grad()
            y_pred = model(prot_x.to(device), f0agv.to(device),
                           evo_x.to(device), length.to(device))

            y_pred_packed = torch.nn.utils.rnn.pack_padded_sequence(
                y_pred, length.cpu(), batch_first=True)
            data_y_packed = torch.nn.utils.rnn.pack_padded_sequence(
                data_y, length, batch_first=True).to(device)

            if data_y_packed.data.numel() > 0:
                loss = fcloss(y_pred_packed.data, data_y_packed.data)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_train_loss += loss.item()
                nb_train += 1
                preds = torch.softmax(y_pred_packed.data, dim=1).argmax(dim=1)
                train_correct += (preds == data_y_packed.data).sum().item()
                train_total += data_y_packed.data.size(0)

        avg_train_loss = epoch_train_loss / max(nb_train, 1)
        train_acc = train_correct / max(train_total, 1)

        # ==================== Validation phase + 8 metrics ====================
        model.eval()
        val_loss, val_correct, val_total, nb_val = 0.0, 0, 0, 0
        val_probs, val_labels = [], []
        with torch.no_grad():
            for prot_x, f0agv, evo_x, data_y, length in val_loader:
                y_pred = model(prot_x.to(device), f0agv.to(device),
                               evo_x.to(device), length.to(device))

                y_pred_packed = torch.nn.utils.rnn.pack_padded_sequence(
                    y_pred, length.cpu(), batch_first=True)
                prob = torch.softmax(y_pred_packed.data, dim=1)[:, 1]
                label = torch.nn.utils.rnn.pack_padded_sequence(
                    data_y, length, batch_first=True).data.to(device)

                val_probs.extend(prob.cpu().numpy())
                val_labels.extend(label.cpu().numpy())

                if label.numel() > 0:
                    loss = fcloss(y_pred_packed.data, label)
                    val_loss += loss.item()
                    nb_val += 1
                    preds = prob.round().int()
                    val_correct += (preds == label).sum().item()
                    val_total += label.size(0)

        avg_val_loss = val_loss / max(nb_val, 1)
        val_acc = val_correct / max(val_total, 1)

        # 8 metrics
        val_probs = np.array(val_probs)
        val_labels = np.array(val_labels)
        val_preds = (val_probs >= 0.5).astype(int)

        tn, fp, fn, tp = metrics.confusion_matrix(val_labels, val_preds).ravel()
        acc = metrics.accuracy_score(val_labels, val_preds)
        bacc = metrics.balanced_accuracy_score(val_labels, val_preds)
        mcc = metrics.matthews_corrcoef(val_labels, val_preds)
        auc = metrics.roc_auc_score(val_labels, val_probs)
        prauc = metrics.average_precision_score(val_labels, val_probs)
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        f1 = metrics.f1_score(val_labels, val_preds)

        scheduler.step(avg_val_loss)

        # ==================== Logging & Early stopping ====================
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}  |  "
                  f"Train Loss: {avg_train_loss:.4f}  Acc: {train_acc:.4f}  |  "
                  f"Val Loss: {avg_val_loss:.4f}  Acc: {val_acc:.4f}")
            print("[Validation 8 metrics] "
                  f"Acc={acc:.4f}  "
                  f"BalAcc={bacc:.4f}  "
                  f"MCC={mcc:.4f}  "
                  f"AUC={auc:.4f}  "
                  f"PR-AUC={prauc:.4f}  "
                  f"Sen={sens:.4f}  "
                  f"Spe={spec:.4f}  "
                  f"F1={f1:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), "PIP_Gen_best.pkl")
            print(f"  Best model saved! Val Loss: {best_val_loss:.4f}")
        else:
            counter += 1
            if counter >= patience:
                print(f"  Early stopping triggered at epoch {epoch + 1}")
                break

    torch.save(model.state_dict(), "PIP_Gen_final.pkl")
    return best_val_loss, val_acc


# Test function
def test():
    # Initialize test dataset and loader
    test_set = BioinformaticsDataset(prot_test, fusion_test, mode='test')
    test_load = DataLoader(
        dataset=test_set,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=coll_paddding
    )

    # Load model
    model = PIPModule()
    model = model.to(device)
    print("========================== Test Results ================================")
    model.load_state_dict(torch.load(r'D:\python code\PIP-Gen\classification_module\code\model\PIP-Gen_best.pkl'))
    model.eval()

    arr_probs = []
    arr_labels = []

    with torch.no_grad():
        for prot_x, f0agv, evo_x, data_y, length in test_load:
            # Forward pass
            y_pred = model(
                prot_x.to(device),
                f0agv.to(device),
                evo_x.to(device),
                length.to(device))

            # Handle variable-length sequences
            y_pred_packed = torch.nn.utils.rnn.pack_padded_sequence(
                y_pred,
                length.to('cpu'),
                batch_first=True
            ).data

            # Compute probabilities
            y_pred = torch.softmax(y_pred_packed, dim=1)
            arr_probs.extend(y_pred[:, 1].cpu().numpy())

            # Get true labels
            data_y_packed = torch.nn.utils.rnn.pack_padded_sequence(
                data_y,
                length,
                batch_first=True
            ).data
            arr_labels.extend(data_y_packed.cpu().numpy())

    # Find optimal threshold
    def find_optimal_threshold(labels, probs):
        thresholds = np.linspace(0.1, 0.9, 100)
        best_f1 = 0
        best_thresh = 0.5

        for thresh in thresholds:
            preds = (probs > thresh).astype(int)
            f1 = metrics.f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    optimal_thresh = find_optimal_threshold(arr_labels, arr_probs)
    print(f"Optimal threshold: {optimal_thresh:.4f}")

    # Predict using optimal threshold
    arr_labels_hyps = (np.array(arr_probs) > optimal_thresh).astype(int)

    # Compute evaluation metrics
    auc = metrics.roc_auc_score(arr_labels, arr_probs)
    acc = metrics.accuracy_score(arr_labels, arr_labels_hyps)
    balanced_acc = metrics.balanced_accuracy_score(arr_labels, arr_labels_hyps)
    mcc = metrics.matthews_corrcoef(arr_labels, arr_labels_hyps)
    ap = metrics.average_precision_score(arr_labels, arr_probs)

    tn, fp, fn, tp = metrics.confusion_matrix(arr_labels, arr_labels_hyps).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1score = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    youden = sensitivity + specificity - 1

    # Precision-Recall curve
    precision_vals, recall_vals, _ = metrics.precision_recall_curve(arr_labels, arr_probs)
    pr_auc = metrics.auc(recall_vals, precision_vals)

    # Output results
    metrics_dict = {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'MCC': mcc,
        'AUC': auc,
        'PR_AUC': pr_auc,
        'AP': ap,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'TP': tp,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1score,
        'Youden Index': youden,
        'Optimal Threshold': optimal_thresh,
        'Positive Ratio': np.mean(arr_labels_hyps)
    }

    print("\n==================== Performance Metrics ====================")
    for key, value in metrics_dict.items():
        print(f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}')

    # Save results
    df = pd.DataFrame([metrics_dict])
    df.to_csv('test_results.csv', index=False)
    print('Test results saved to test_results.csv')

    # Save probabilities and labels for further analysis
    prob_label_df = pd.DataFrame({
        'prob': arr_probs,
        'label': arr_labels,
        'pred': arr_labels_hyps
    })
    prob_label_df.to_csv('prob_label.csv', index=False)
    print('Probabilities and labels saved to prob_label.csv')
    return acc, mcc


if __name__ == "__main__":
    # Set random seed
    set_seed(42)

    # CUDA availability check
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"Device: {device}")

    # Dataset paths
    fusion_train = [r'D:\python code\PIP-Gen\classification_module\PIP Feature\PIP-Train-ESM480..csv']
    prot_train = [r'D:\python code\PIP-Gen\classification_module\PIP Feature\PIP-Train-ESM480..csv']
    fusion_test = [r'D:\python code\PIP-Gen\classification_module\PIP Feature\PIP-Test-ESM480..csv']
    prot_test = [r'D:\python code\PIP-Gen\classification_module\PIP Feature\PIP-Test-ESM480..csv']

    # Train model (uncomment this line)
    #train()

    # Test model (comment or remove)
    test()