[Uploading README.md…]()
# PIP-Gen: Proinflammatory Peptide Generation via Latent Diffusion

PIP-Gen is a deep generative model for designing novel **proinflammatory peptides**. It employs a **latent diffusion model** operating in the embedding space of **ESM-2** (a protein language model), then decodes the generated embeddings back into amino-acid sequences. A companion **classification module** predicts whether candidate peptides have proinflammatory/antimicrobial activity for downstream validation and selection.

> Instead of generating peptide tokens autoregressively, PIP-Gen learns to reverse a diffusion process applied to ESM-2 hidden-state embeddings — producing diverse, realistic peptide sequences via DDIM sampling with classifier-free guidance.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Training Pipeline                      │
├──────────────────────────────────────────────────────────┤
│  Proinflammatory   ──►  ESM-2 (8M)   ──►  Embeddings     │
│  Peptide Sequences       Encoder         [L, 320]        │
│                                 │                         │
│                    ┌────────────▼────────────┐           │
│                    │  1D U-Net Denoiser      │           │
│                    │  - 3 encoder blocks     │           │
│                    │  - 3 decoder blocks     │           │
│                    │  - FiLM conditioning    │           │
│                    │  - Class label embed    │           │
│                    │  - Time embedding       │           │
│                    └────────────┬────────────┘           │
│                                 │                         │
│                    ┌────────────▼────────────┐           │
│                    │  ESM-2 lm_head          │           │
│                    │  (embedding → token)     │           │
│                    └────────────┬────────────┘           │
│                                 ▼                         │
│                          Amino-Acid Sequence              │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│                   Generation (Inference)                  │
├──────────────────────────────────────────────────────────┤
│  Random Noise   ──►  DDIM Sampling (50 steps)            │
│  [L, 320]            + Classifier-Free Guidance (cfs=0.8) │
│                              │                            │
│                              ▼                            │
│                       Denoised Embedding                  │
│                              │                            │
│                    ┌─────────▼─────────┐                  │
│                    │  Trained Decoder   │                  │
│                    │  (lm_head)         │                  │
│                    └─────────┬─────────┘                  │
│                              ▼                            │
│                       Novel Peptide Sequence              │
│                       (top-k/top-p + 4-gram penalty)      │
└──────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|---|---|
| **1D U-Net Denoiser** | Operates on `[batch, length, 320]` embeddings. 3 encoder/decoder stages with residual blocks, FiLM conditioning on timestep and class label, and sinusoidal time embeddings. |
| **Label Embedder** | 3-class embedding with dropout (0.2) for classifier-free guidance — class 3 serves as the unconditional/dropout token. |
| **Noise Schedule** | 500 timesteps, custom `alpha_cumprod = 1 - sqrt(t/(T + sqrt_s))`. |
| **Decoder (lm_head)** | Fine-tuned ESM-2 language model head that maps embeddings back to token IDs via cross-entropy. |
| **DDIM Sampler** | 50-step deterministic sampling with classifier-free guidance, length distribution matching, and 4-gram repetition penalty. |
| **PIP Classifier** | Channel attention → biGRU → 1D CNN backbone with a specificity branch. Uses Focal Loss for class imbalance and evaluates 8 metrics. |

## Directory Structure

```
PIP-Gen/
├── config.ini                       # Central hyperparameter configuration
├── UnetDiff/                        # Diffusion model package
│   ├── models/
│   │   ├── UNetDenoiser.py            # 1D U-Net denoiser
│   │   └── layers/
│   │       ├── EMA.py                 # Exponential Moving Average
│   │       ├── LabelEmbedder.py       # Class label embedding + CFG dropout
│   │       └── SinusoidalPositionEmbeddings.py  # Time embedding
│   └── utils/
│       ├── DiffDataset.py             # Sequence/label dataset loaders
│       └── utils.py                   # Noise extraction, cosine sim, utilities
├── classification_module/           # Proinflammatory/AMP classifier
│   ├── PIP Feature/                   # Precomputed ESM-480 features
│   └── code/
│       ├── feature extraction/
│       │   └── ESM extraction.py      # ESM-2 feature extraction
│       └── model/
│           ├── PIP-Gen.py             # PIPModule classifier + train/test
│           └── LossFunction/
│               └── focalLoss.py       # Focal loss for class imbalance
├── prediction/                      # Bundled ESM-2 checkpoints
│   ├── ESM2-8M/                       # esm2_t6_8M_UR50D (hidden=320)
│   └── esm2_t12_35M_UR50D/            # 35M model (hidden=480)
├── data/
│   ├── Generation Data/               # Diffusion training/validation data
│   └── PIP Data/                      # AMP peptide classification data
├── scripts/
│   ├── train_denoiser.py              # Stage 1: Train the U-Net denoiser
│   ├── train_decoder.py               # Stage 2: Train the lm_head decoder
│   ├── sample_randomLen.py            # Stage 3: Generate novel peptides
│   └── logs/                          # TensorBoard event files
└── save_model/                      # Trained model checkpoints
    ├── denoiser_model.pkl
    └── proinflammatory_decoder_best.pkl
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Raccon-CMD/PIP-Gen.git
   cd PIP-Gen
   ```

2. **Install dependencies:**

   ```bash
   pip install torch torchvision transformers timm scikit-learn pandas numpy biopython tqdm colorama tensorboard
   ```

3. **ESM-2 model checkpoints** are bundled in `prediction/` — no download needed.
   - `prediction/ESM2-8M/` — esm2_t6_8M_UR50D (base model for diffusion)
   - `prediction/esm2_t12_35M_UR50D/` — 35M model for classification features

4. **Update paths in `config.ini`** to match your local setup. The config currently contains hardcoded Windows paths that need adjustment.

## Usage

### Training Pipeline (3 Stages)

**Stage 1 — Train the Denoiser:**

```bash
python scripts/train_denoiser.py
```

Trains the 1D U-Net to predict clean ESM-2 embeddings from noisy inputs. Checkpoints saved to `save_model/`.

**Stage 2 — Train the Decoder:**

```bash
python scripts/train_decoder.py
```

Fine-tunes the ESM-2 `lm_head` to map embeddings back to amino-acid tokens. Saves best checkpoint by validation loss.

**Stage 3 — Generate Novel Peptides:**

```bash
python scripts/sample_randomLen.py
```

Runs DDIM sampling with classifier-free guidance to generate ~3000 novel proinflammatory peptide sequences. Output written to FASTA format in `scripts/sample_proinflammatory_ddim/`.

### Classification

**Extract ESM-2 features:**

```bash
python classification_module/code/feature\ extraction/ESM\ extraction.py
```

**Train or evaluate the classifier:**

```bash
python classification_module/code/model/PIP-Gen.py
```

> The `train()` function is commented out by default; `test()` runs on launch and loads the best checkpoint.

## Configuration

All hyperparameters are centralized in `config.ini` under the `[UnetDiff_conf]` section:

| Parameter | Description | Default |
|---|---|---|
| `EMBEDDING_DIM` | ESM-2 hidden dimension | 320 |
| `NUM_CLASSES` | Number of class labels | 3 |
| `TIME_STEPS` | Diffusion timesteps | 500 |
| `CHANNELS` | U-Net channel count | 128 |
| `BATCH_SIZE` | Training batch size | 64 |
| `EPOCHES` | Training epochs | 2000 |
| `LEARNING_RATE` | Optimizer learning rate | 1e-3 |
| `EMA_DECAY` | EMA decay rate | 0.99 |
| `CLASSIFIER_FREE_SCALE` | CFG scale for sampling | 0.8 |

## Data

### Diffusion Data

| Dataset | Sequences | Format |
|---|---|---|
| Train | 1,245 | CSV (`sequence, label`) |
| Validation | 171 | CSV (`sequence, label`) |

### Classification Data

| Dataset | Samples | Format |
|---|---|---|
| PIP Train | 2,872 | CSV (`aa_seq, aa_len, AMP`) |
| PIP Test | 342 | CSV |

### Label Convention
- `label = 1` → proinflammatory peptide
- `label = 0` → non-proinflammatory
- `label = 3` → unconditional (used for classifier-free guidance dropout)

## Model Checkpoints

| Model | Path | Size | Description |
|---|---|---|---|
| ESM-2 8M | `prediction/ESM2-8M/` | ~30 MB | Base encoder for diffusion |
| ESM-2 35M | `prediction/esm2_t12_35M_UR50D/` | ~130 MB | Encoder for classification features |
| Denoiser (U-Net) | `save_model/denoiser_model.pkl` | ~31 MB | Trained 1D U-Net denoiser |
| Decoder Head | `save_model/proinflammatory_decoder_best.pkl` | ~452 KB | Trained lm_head decoder |
| PIP Classifier | `classification_module/code/model/PIP-Gen_best.pkl` | ~43 MB | Best classifier checkpoint |

## Key Design Choices

- **Classifier-Free Guidance** (`cfs=0.8`): Balances sample quality and diversity without a separate classifier model.
- **DDIM Sampling** (50 steps, η=0): Deterministic reverse process for fast, reproducible generation.
- **4-gram Repetition Penalty**: Prevents degenerate repetitive sequence generation during decoding.
- **Length Distribution Matching**: Generated sequence lengths are sampled from the empirical training-set length distribution.
- **Focal Loss**: Addresses class imbalance in the antimicrobial peptide classifier (α=[0.25, 0.75], γ=2).
- **EMA**: Exponential moving average of model weights (decay=0.99) stabilizes training with small datasets.

## Notes

- Paths in configuration files and scripts currently use **hardcoded absolute Windows paths** — update these for your environment.
- The `.gitignore` excludes only `save_model/checkpoint.pth` (resumable training state); model checkpoints and `__pycache__/` directories are tracked.
- `prediction/esm2_t12_35M_UR50D` is a **nested git repository** with LFS-managed model files.
- Some legacy file references point to a separate `DeepAIP` project and can be safely ignored.

## License

This project is for research purposes. Please see the repository for license details.

---

🤖 *README generated with assistance from Claude Code*
