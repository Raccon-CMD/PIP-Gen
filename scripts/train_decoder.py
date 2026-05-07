import torch
import random
import configparser
import numpy as np
import pandas as pd
import torch.nn.functional as F
from colorama import Fore, Style
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel, AutoModelForMaskedLM
from UnetDiff.utils.DiffDataset import XDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import device
from torch.cuda import is_available


DEVICE = device("cuda:0" if is_available() else "cpu")
print("DEVICE:", DEVICE)

conf = configparser.ConfigParser()
conf.optionxform = str
config_path = r'D:\python code\PIP-Gen\config.ini'

encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin-1']
config_read = False
for encoding in encodings:
    try:
        conf.read(config_path, encoding=encoding)
        if conf.has_section('UnetDiff_conf'):
            print(f"Successfully read config with encoding: {encoding}")
            config_read = True
            break
    except Exception as e:
        print(f"Failed with encoding {encoding}: {e}")

if not config_read:
    print("Error: Cannot read config file, please check encoding.")
    exit(1)

conf_dict = dict(conf.items('UnetDiff_conf'))

def repetition_penalty_ngram(logits, seq_ids, n=4, penalty=1.5):
    batch_size, seq_len, vocab = logits.shape
    for i in range(n, seq_len):
        for b in range(batch_size):
            recent = seq_ids[b, i - n:i]
            for token in recent:
                logits[b, i, token] /= penalty
    return logits


def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    batch_size, seq_len, vocab_size = logits.shape
    sampled_ids = []
    for b in range(batch_size):
        seq_tokens = []
        for t in range(seq_len):
            logit = logits[b, t, :] / temperature
            probs = F.softmax(logit, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            sorted_probs /= sorted_probs.sum()
            sampled_token = sorted_indices[torch.multinomial(sorted_probs, num_samples=1).item()]
            seq_tokens.append(sampled_token.item())
        sampled_ids.append(seq_tokens)
    return torch.tensor(sampled_ids, dtype=torch.long, device=logits.device)


def evaluate_decoder(decoder, esm2_model, tokenizer, val_loader, device):
    decoder.eval()
    esm2_model.eval()
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for datas in val_loader:
            seq_encode = tokenizer(
                datas['sequences'],
                max_length=int(conf_dict['MAX_LENGTH']) + 2,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            seq_ids = seq_encode['input_ids'].to(device)
            mask = seq_encode['attention_mask'].to(device)

            x0 = esm2_model(seq_ids, mask).last_hidden_state
            logits = decoder(x0)
            V = logits.size(-1)

            ce_loss = F.cross_entropy(
                logits.view(-1, V),
                seq_ids.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='none'
            ).view_as(seq_ids)
            ce_loss = (ce_loss * mask).sum() / mask.sum()

            total_loss += ce_loss.item() * seq_ids.size(0)
            total_samples += seq_ids.size(0)

    return total_loss / total_samples

def train_decoder_improved(data_type='proinflammatory', epochs=30, patience=5):
    LOCAL_ESM = r'D:\python code\PIP-Gen\prediction\ESM2-8M'

    tokenizer = EsmTokenizer.from_pretrained(LOCAL_ESM, trust_remote_code=True)
    esm2_model = EsmModel.from_pretrained(LOCAL_ESM, trust_remote_code=True).to(DEVICE)
    decoder = AutoModelForMaskedLM.from_pretrained(LOCAL_ESM, trust_remote_code=True).lm_head.to(DEVICE)

    train_dataset = XDataset(pd.read_csv('D:\python code\PIP-Gen\data\Generation Data\Gen_proinflammatory_Train.csv'))
    val_dataset = XDataset(pd.read_csv('D:\python code\PIP-Gen\data\Generation Data\Gen_proinflammatory-Val.csv'))

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=512, pin_memory=True, shuffle=True,
        persistent_workers=True, num_workers=8
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=512, pin_memory=True, shuffle=False,
        persistent_workers=True, num_workers=8
    )

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    writer = SummaryWriter(log_dir='./logs/decoder')

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Start training, total {epochs} epochs")

    for epoch in range(epochs):
        pbar = tqdm(total=len(train_data_loader), desc=f'Epoch {epoch + 1}/{epochs}')
        decoder.train()
        epoch_losses = []

        for datas in train_data_loader:
            seq_encode = tokenizer(
                datas['sequences'],
                max_length=int(conf_dict['MAX_LENGTH']) + 2,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
            seq_ids_list, train_attention_mask = seq_encode['input_ids'].to(DEVICE), seq_encode['attention_mask'].to(DEVICE)

            with torch.no_grad():
                x0 = esm2_model(seq_ids_list, train_attention_mask).last_hidden_state

            pred_logits = decoder(x0)

            optimizer.zero_grad()
            ce_losses = F.cross_entropy(
                pred_logits.view(-1, pred_logits.shape[-1]),
                seq_ids_list.view(-1),
                reduction='none'
            )
            ce_losses = ce_losses * train_attention_mask.reshape(-1)
            loss = torch.sum(ce_losses) / torch.sum(train_attention_mask)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.update(1)
            pbar.set_description_str(
                f"Epoch {epoch + 1}/{epochs} " + Fore.RED + f"loss:{loss.item():.4f}" + Style.RESET_ALL)
            writer.add_scalar("train/step_loss", loss.item(), epoch * len(train_data_loader) + pbar.n)

        pbar.close()
        avg_train_loss = np.mean(epoch_losses)

        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for datas in val_data_loader:
                seq_encode = tokenizer(
                    datas['sequences'],
                    max_length=int(conf_dict['MAX_LENGTH']) + 2,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt"
                )
                seq_ids_list, val_attention_mask = seq_encode['input_ids'].to(DEVICE), seq_encode['attention_mask'].to(DEVICE)

                x0 = esm2_model(seq_ids_list, val_attention_mask).last_hidden_state
                pred_logits = decoder(x0)

                ce_losses = F.cross_entropy(
                    pred_logits.view(-1, pred_logits.shape[-1]),
                    seq_ids_list.view(-1),
                    reduction='none'
                )
                ce_losses = ce_losses * val_attention_mask.reshape(-1)
                val_loss = torch.sum(ce_losses) / torch.sum(val_attention_mask)
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)

        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("val/epoch_loss", avg_val_loss, epoch + 1)
        writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch + 1)

        print(f"Epoch {epoch + 1}: train loss = {avg_train_loss:.4f}, val loss = {avg_val_loss:.4f}")

        # 只保存最佳模型，不保存每个epoch的检查点
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(decoder.state_dict(), f"../save_model/{data_type}_decoder_best.pkl")
            print(f"✅ Best model saved, val loss: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⚠️ Validation loss did not improve, early stop counter: {patience_counter}/{patience}")

        # 删除了原来保存每个epoch检查点的代码：torch.save(decoder.state_dict(), f"../save_model/{data_type}_decoder_epoch{epoch + 1}.pkl")

        scheduler.step()

        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered, training finished.")
            break

    writer.close()
    print(f"Training completed, best val loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    torch.manual_seed(int(conf_dict['SEED']))
    torch.cuda.manual_seed_all(int(conf_dict['SEED']))
    np.random.seed(int(conf_dict['SEED']))
    random.seed(int(conf_dict['SEED']))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_decoder_improved(data_type='proinflammatory', epochs=30, patience=5)