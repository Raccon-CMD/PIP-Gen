import os, random, numpy as np, pandas as pd, torch, torch.nn.functional as F
import configparser, math
from Bio import SeqIO
from tqdm import tqdm
from UnetDiff.models.UNetDenoiser import UNetDenoiser
from UnetDiff.utils.DiffDataset import XYDataset
from UnetDiff.utils.utils import set_seed, extract
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import Counter
from torch import device
from torch.cuda import is_available

sample_type = 'proinflammatory'
cfs = 0.8
n = 1
target_total = 3000
batch_size = 100
device = device("cuda:0" if is_available() else "cpu")
conf = configparser.ConfigParser()
conf.read(r'D:\python code\PIP-Gen\config.ini', encoding='utf-8')
conf_dict = dict(conf.items('UnetDiff_conf'))
set_seed(int(conf_dict['seed']))
labels = {'proinflammatory': 1}
peptide_file = 'proinflammatory'

ddim_steps = 50
ddim_eta = 0.0

class LengthSampler:
    def __init__(self, path, max_len=254):
        data = [str(r.seq) for r in SeqIO.parse(path, "fasta")]
        self.dataset_len = np.clip([len(t) for t in data], 0, max_len)
        freqs = Counter(self.dataset_len)
        self.distrib = np.array([freqs.get(i, 0) for i in range(max_len + 1)])
        self.distrib = self.distrib / self.distrib.sum()

    def sample(self, num_samples):
        return np.argmax(np.random.multinomial(1, self.distrib, size=num_samples), axis=1)

# ---------- top-k/top-p ----------
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_with_penalty(logits, temperature=2.5, top_k=20, top_p=0.95, penalty=1.5,
                        eos_token_id=2, min_eos_prob=0.05):
    B, L, V = logits.shape
    sampled = []
    for b in range(B):
        tokens = []
        for t in range(L):
            logit = logits[b, t] / temperature
            # repetition penalty on last 4 tokens
            for prev in tokens[-4:]:
                logit[prev] /= penalty
            # ensure eos gets at least min_eos_prob
            eos_logit = logit[eos_token_id]
            if eos_logit < math.log(min_eos_prob):
                logit[eos_token_id] = math.log(min_eos_prob)
            filtered = top_k_top_p_filtering(logit, top_k=top_k, top_p=top_p)
            prob = F.softmax(filtered, dim=-1)
            tok = torch.multinomial(prob, 1).item()
            tokens.append(tok)
            if tok == eos_token_id:  # early stop
                break

        while len(tokens) < L:
            tokens.append(eos_token_id)
        sampled.append(tokens)
    return torch.tensor(sampled, dtype=torch.long, device=logits.device)

def ddim_sample(denoiser, x_t_shape, timesteps, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                len_list=None, use_attention=True, cfs=1.,
                peptide_type='proinflammatory', index=1, ddim_steps=50, ddim_eta=0.0):

    unconditional = torch.zeros(x_t_shape[0], device=device, dtype=torch.long) + 3
    conditional = torch.zeros(x_t_shape[0], device=device, dtype=torch.long) + labels[peptide_type]

    if len_list is None:
        ones_counts = LengthSampler(rf'D:\python code\PIP-Gen\data\PIP Data\{peptide_file}.fasta',
                                    int(conf_dict['max_length'])).sample(x_t_shape[0]) + 2
    else:
        ones_counts = np.array(len_list) + 2
    attn_mask = np.zeros((len(ones_counts), int(conf_dict['max_length']) + 2), dtype=int)
    for i, c in enumerate(ones_counts):
        attn_mask[i, :c] = 1
    attn_mask = torch.from_numpy(attn_mask).to(device)

    x_t = torch.randn(x_t_shape, device=device)

    if ddim_steps < timesteps:
        step_indices = torch.linspace(0, timesteps - 1, ddim_steps, dtype=torch.long, device=device)
    else:
        step_indices = torch.arange(0, timesteps, dtype=torch.long, device=device)

    timestep_sequence = (timesteps - 1 - step_indices).tolist()
    timestep_sequence = sorted(timestep_sequence, reverse=True)

    print(
        f"DDIM sampling: using {ddim_steps} steps (η={ddim_eta}), timestep range: {min(timestep_sequence)}-{max(timestep_sequence)}")

    for i, t in enumerate(tqdm(timestep_sequence, desc=f'DDIM sampling {index}')):

        batch_t = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)

        sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, batch_t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, batch_t, x_t.shape)
        alphas_cumprod_t = extract(alphas_cumprod, batch_t, x_t.shape)

        with torch.no_grad():
            cond_pred = denoiser(x_t, batch_t, y=conditional, attention_mask=attn_mask)
            uncond_pred = denoiser(x_t, batch_t, y=unconditional, attention_mask=attn_mask)
        pred_x0 = (1 + cfs) * cond_pred - cfs * uncond_pred

        noise_scale = 0.3
        pred_x0 = pred_x0 + torch.randn_like(pred_x0) * noise_scale

        if t == 0:
            x_t = pred_x0
            continue

        pred_noise = (x_t - sqrt_alphas_cumprod_t * pred_x0) / sqrt_one_minus_alphas_cumprod_t

        t_prev = timestep_sequence[i + 1] if i + 1 < len(timestep_sequence) else 0
        batch_t_prev = torch.full((x_t.shape[0],), t_prev, device=device, dtype=torch.long)

        sqrt_alphas_cumprod_prev = extract(sqrt_alphas_cumprod, batch_t_prev, x_t.shape)
        sqrt_one_minus_alphas_cumprod_prev = extract(sqrt_one_minus_alphas_cumprod, batch_t_prev, x_t.shape)
        alphas_cumprod_prev = extract(alphas_cumprod, batch_t_prev, x_t.shape)

        x_t_prev_det = sqrt_alphas_cumprod_prev * pred_x0 + sqrt_one_minus_alphas_cumprod_prev * pred_noise

        if ddim_eta > 0:
            sigma_t = ddim_eta * torch.sqrt(
                torch.clamp(
                    (1 - alphas_cumprod_prev) / (1 - alphas_cumprod_t) *
                    (1 - alphas_cumprod_t / alphas_cumprod_prev),
                    min=1e-8
                )
            )
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev_det + sigma_t * noise
        else:
            x_t_prev = x_t_prev_det

        x_t = x_t_prev

    print("pred_x0 std along batch:", pred_x0.std(dim=0).mean().item())
    return pred_x0


# ---------- Model loading ----------
denoiser_mlp = [int(i) for i in conf_dict['denoiser_mlp'].split(',')]
denoiser = UNetDenoiser('../' + conf_dict['denoiser_esm_model_name'],
                        int(conf_dict['denoiser_embedding']), denoiser_mlp).to(device)
denoiser.load_state_dict(
    torch.load(r"D:\python code\PIP-Gen\save_model\denoiser_model.pkl", map_location=device),
    strict=False
)
denoiser.eval()

tokenizer = AutoTokenizer.from_pretrained(r'D:\python code\PIP-Gen\prediction\ESM2-8M', trust_remote_code=True)
esm2_model = AutoModelForMaskedLM.from_pretrained(r'D:\python code\PIP-Gen\prediction\ESM2-8M', trust_remote_code=True).to(device)
decoder = esm2_model.lm_head
decoder.load_state_dict(
    torch.load(r"D:\python code\PIP-Gen\save_model\proinflammatory_decoder_best.pkl", map_location=device))
esm2_model.eval()
decoder.eval()

# ---------- Noise schedule ----------
timesteps = int(conf_dict['time_steps'])
t = torch.arange(0, timesteps, device=device, dtype=torch.long)
alphas_cumprod = 1 - torch.sqrt((t + 1) / (timesteps + float(conf_dict['sqrt_s'])))

betas = []
for i in range(timesteps):
    if i == 0:
        betas.append(1 - alphas_cumprod[i])
    else:
        betas.append(1 - (alphas_cumprod[i] / alphas_cumprod[i - 1]))
betas = torch.stack(betas).to(device)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

print(f"Noise schedule parameters:")
print(f"  alphas_cumprod range: [{alphas_cumprod[0].item():.6f}, {alphas_cumprod[-1].item():.6f}]")
print(f"  sqrt_alphas_cumprod range: [{sqrt_alphas_cumprod[0].item():.6f}, {sqrt_alphas_cumprod[-1].item():.6f}]")

# ---------- Sampling + loop ----------
train_dataset = XYDataset(pd.read_csv(r'D:\python code\PIP-Gen\data\Generation Data\Gen_proinflammatory_Train.csv'))
vocab_dict = {v: k for k, v in tokenizer.get_vocab().items()}
seq_list = []
output_dir = './sample_proinflammatory_ddim'
os.makedirs(output_dir, exist_ok=True)

print(f"Start DDIM sampling, target {target_total} sequences ...")
print(f"DDIM parameters: {ddim_steps} steps, η={ddim_eta}")
print(f"Original training steps: {timesteps}")

# test a small batch first
test_batch_size = min(batch_size, 10)
print(f"Testing {test_batch_size} samples ...")

with torch.no_grad():
    x0_test = ddim_sample(denoiser,
                          [test_batch_size, int(conf_dict['max_length']) + 2, int(conf_dict['denoiser_embedding'])],
                          timesteps, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                          cfs=cfs, use_attention=True, peptide_type=peptide_file, index=1,
                          ddim_steps=ddim_steps, ddim_eta=ddim_eta)

    print(f"Test successful! x0 shape: {x0_test.shape}")

# formal sampling
while len(seq_list) < target_total:
    need = target_total - len(seq_list)
    current_batch = min(batch_size, need)
    print(f"Need {need} more, sampling {current_batch} ...")

    with torch.no_grad():
        x0 = ddim_sample(denoiser,
                         [current_batch, int(conf_dict['max_length']) + 2, int(conf_dict['denoiser_embedding'])],
                         timesteps, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                         cfs=cfs, use_attention=True, peptide_type=peptide_file, index=1,
                         ddim_steps=ddim_steps, ddim_eta=ddim_eta)

        pred_score = decoder(x0)
        seq_ids_list = sample_with_penalty(pred_score, eos_token_id=2, min_eos_prob=0.05)

        reason = {'no_cls': 0, 'special': 0, 'no_eos': 0, 'low_complexity': 0, 'ok': 0}
        for seq_ids in seq_ids_list:
            seq_ids = seq_ids.cpu()
            if seq_ids[0] != 0:
                reason['no_cls'] += 1
                continue
            if torch.any(torch.isin(seq_ids, torch.tensor([1, 3]))):
                reason['special'] += 1
                continue
            eos_idx = (seq_ids == 2).nonzero(as_tuple=True)[0]
            if eos_idx.numel() == 0:
                reason['no_eos'] += 1
                continue
            seq = tokenizer.decode(seq_ids[1:eos_idx[0]]).replace(" ", "").replace("<cls>", "").replace("<eos>",
                                                                                                        "").replace(
                "<pad>", "")
            if len(seq) == 0 or len(set(seq)) / len(seq) < 0.25:
                reason['low_complexity'] += 1
                continue
            reason['ok'] += 1
            seq_list.append(seq)

    print(f"Batch filter stats: {reason} | Current total {len(seq_list)} sequences")

    # save temporary checkpoint every 500
    if len(seq_list) % 500 == 0:
        temp_file = f"{output_dir}/temp_{len(seq_list)}.fasta"
        with open(temp_file, 'w') as f:
            for idx, seq in enumerate(seq_list[-500:], len(seq_list) - 499):
                seq = seq.replace("X", random.choice("ACDEFGHIKLMNPQRSTVWY"))
                f.write(f">generated_peptide_{idx}\n{seq}\n")
        print(f"Temporary saved to {temp_file}")

print(f"DDIM sampling finished, total {len(seq_list)} sequences saved.")
with open(f"{output_dir}/unet_proinflammatory_peptides_3000_ddim.fasta", 'w') as f:
    for idx, seq in enumerate(seq_list, 1):
        seq = seq.replace("X", random.choice("ACDEFGHIKLMNPQRSTVWY"))
        f.write(f">generated_peptide_{idx}\n{seq}\n")

print("\n=== DDIM sampling completed ===")
print(f"DDIM steps: {ddim_steps} (original training steps: {timesteps})")
print(f"Speedup ratio: {timesteps / ddim_steps:.1f}x")
print(f"Saved to: {output_dir}/unet_proinflammatory_peptides_3000_ddim.fasta")