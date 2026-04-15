# Energy-Based Model Prior for Latent Variable Models

A PyTorch implementation of **learning an Energy-Based Model (EBM) prior** over the latent space of generative models, based on the paper:

> *Learning Multimodal Latent Generative Models with Energy-Based Prior* — included in this repo as PDF.

The core idea is to replace the standard Gaussian prior $p_0(z)$ in a VAE with a learned, expressive prior $p_\theta(z) \propto e^{-E_\theta(z)} \cdot p_0(z)$, trained via **Contrastive Divergence (CD)** and **Unadjusted Langevin Algorithm (ULA)** MCMC.

---

## 📐 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  Generative Pipeline                  │
│                                                      │
│  Input Text                                          │
│     │                                                │
│     ▼                                                │
│  ┌─────────┐    mu, logvar     ┌───────────────────┐ │
│  │ Encoder │ ─────────────────▶│  Reparameterize   │ │
│  └─────────┘                  └────────┬──────────┘ │
│                                        │ z ~ q(z|x)  │
│                                        ▼             │
│  ┌──────────────────────────────────────────────┐    │
│  │          FlatEBM  E_θ(z)                     │    │
│  │   Linear → SiLU → [ResBlock × 4] → Linear   │    │
│  │     (Spectral Norm on all linear layers)     │    │
│  └──────────────────────────────────────────────┘    │
│                                        │             │
│                                        ▼             │
│  ┌─────────┐ ◀── z ──────────────────────────────   │
│  │ Decoder │ ──▶  Reconstructed Text               │ │
│  └─────────┘                                        │
└──────────────────────────────────────────────────────┘
```

---

## 🗂️ Repository Structure

```
ebm_prior/
│
├── unimodal_ebm.py       # Core model: FlatEBM, MCMC_Sampler,
│                         #   UnimodalGenerativeModelWithEBMPrior,
│                         #   and train_step_algorithm1 (Algorithm 1)
│
├── algo2_ebm.py          # Algorithm 2 variant: decoder-only training
│                         #   (no amortized encoder). Includes
│                         #   SmallRandomDecoder, OptimusDecoderWrapper,
│                         #   sample_langevin_prior / posterior
│
├── losses.py             # compute_reconstruction_loss (masked cross-entropy
│                         #   returning per-sentence and per-token losses)
│
├── train.py              # Training script for Algorithm 1 (full VAE + EBM)
│                         #   Uses Optimus encoder + decoder, WandB logging,
│                         #   cyclic beta annealing, and MCMC replay buffer
│
├── train_algo2.py        # Training script for Algorithm 2 (decoder-only)
│                         #   Uses Optimus decoder or SmallRandomDecoder
│
├── logs/                 # Raw training logs from past runs
│
└── Learning Multimodal Latent Generative Models with Energy-Based Prior.pdf
```

---

## ⚙️ How It Works

### Algorithm 1 — Full VAE + EBM Prior

The model alternates between two update phases per batch:

**Phase 1 — Update VAE (Encoder + Decoder)**

$$
\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot \text{KL}_{\text{clamp}}(q_\phi \parallel p_0) + \mathbb{E}_{q_\phi}[E_\theta(z)]
$$

- Reconstruction loss: masked cross-entropy (per-sentence sum, batch mean)
- KL divergence uses **Free Bits** (softplus hinge at margin=100) to prevent posterior collapse
- EBM energy term is unclamped to let the EBM guide the encoder freely

**Phase 2 — Update EBM via Contrastive Divergence**

$$
\mathcal{L}_{\text{CD}} = \mathbb{E}_{q_\phi(z|x)}[E_\theta(z)] - \mathbb{E}_{p_\theta(z)}[E_\theta(z)]
$$

- Positive samples: latent codes from the encoder posterior $z \sim q_\phi(z|x)$
- Negative samples: chains run from a **replay buffer** (95% buffer / 5% fresh noise) via ULA

### Algorithm 2 — Decoder-Only (No Amortized Encoder)

Drops the encoder entirely. Both prior and posterior samples are drawn via Langevin MCMC:

| | Prior Sampling | Posterior Sampling |
|---|---|---|
| **Target** | $p_\theta(z) \propto e^{-E_\theta(z)} p_0(z)$ | $p(z\|x) \propto p_G(x\|z) \cdot p_\theta(z)$ |
| **Steps** | $K_0 = 60$, $\alpha_0 = 0.4$ | $K_1 = 40$, $\alpha_1 = 0.1$ |
| **Gradient** | $\nabla_z E_\theta(z) + z$ | $\nabla_z \mathcal{L}_\text{recon} + \nabla_z E_\theta(z) + z$ |

---

## 🧠 Key Components

### `FlatEBM` (`unimodal_ebm.py`)

A deep MLP that maps a latent vector $z \in \mathbb{R}^{768}$ to a scalar energy:

- Input projection: `Linear(768 → 1024)` with **Spectral Norm**
- 4× `ResidualBlock`: two spectrally-normed linear layers + SiLU activations
- Output: `Linear(1024 → 1)` with **Spectral Norm** (enforces global Lipschitz continuity)

### `MCMC_Sampler` (`unimodal_ebm.py`)

Implements **Unadjusted Langevin Algorithm (ULA)** with a persistent replay buffer:

$$
z_{t+1} = z_t - \frac{\alpha^2}{2}\left(\nabla_z E_\theta(z_t) + z_t\right) + \alpha \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

- Buffer stored on **CPU** to conserve VRAM
- Logs per-step diagnostics: total energy, EBM energy, prior energy, gradient norms, distance from chain start

### `SmallRandomDecoder` (`algo2_ebm.py`)

A lightweight, **randomly initialized** transformer decoder (no pretrained weights):

- Token + positional embeddings → latent injection via element-wise add + LayerNorm
- 4-layer causal `TransformerEncoder` with pre-norm and 8 attention heads
- `hidden_dim=256`, `latent_dim=768` by default

### `OptimusDecoderWrapper` (`algo2_ebm.py`)

Fixes an **attention mask size mismatch** when injecting a latent vector as KV memory into the Optimus GPT-2 decoder. Prepends a column of ones to the mask so it covers the extra virtual "latent token" position created by `past=z`.

---

## 📊 Training Details

| Hyperparameter | Algorithm 1 | Algorithm 2 |
|---|---|---|
| Latent dim | 768 | 768 |
| Batch size | 128 | 64 |
| LR (VAE/Decoder) | 3e-5 | 1e-5 |
| LR (EBM) | 3e-5 | 2e-5 |
| MCMC steps (prior) | 50 | 60 |
| MCMC step size (prior) | 0.01 | 0.4 |
| MCMC steps (posterior) | — | 40 |
| MCMC step size (posterior) | — | 0.1 |
| KL free bits margin | 100 | N/A |
| β annealing | Cyclic (0.005 → 1.0) | Fixed |
| Replay buffer | ✅ 10k, 95% ratio | ❌ |
| Gradient clipping | 1.0 (max norm) | N/A |
| Optimizer | Adam (β=0.5, 0.999) | Adam (β=0.5, 0.999) |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch transformers wandb pandas scikit-learn tqdm
```

> **Note:** The `train.py` and `train_algo2.py` scripts depend on the [Optimus](https://github.com/ChunyuanLI/Optimus) codebase for the pretrained BERT encoder and GPT-2 decoder. Adjust the `inference_dir` and `CHECKPOINT_PATH` variables to point to your local Optimus installation. Alternatively, use `SmallRandomDecoder` from `algo2_ebm.py` for a self-contained run without pretrained weights.

### Run Algorithm 1 (Full VAE + EBM)

```bash
python train.py
```

### Run Algorithm 2 (Decoder-Only EBM)

```bash
python train_algo2.py
```

### Use the Core Modules Directly

```python
from unimodal_ebm import FlatEBM, MCMC_Sampler
from losses import compute_reconstruction_loss

# Instantiate the EBM
ebm = FlatEBM(latent_dim=768, hidden_dim=1024, num_res_blocks=4)

# Run MCMC to draw samples from the prior
sampler = MCMC_Sampler(ebm, latent_dim=768, step_size=0.01, num_steps=50)
z_samples, mcmc_log = sampler.sample(batch_size=32, device='cuda')
```

---

## 📈 Logging & Monitoring

Training is logged to **Weights & Biases**. Key tracked metrics:

- `loss_g` — decoder reconstruction loss (per-sentence mean NLL)
- `token_loss` — per-token cross-entropy (for perplexity reference)
- `loss_e (CD)` — contrastive divergence loss
- `en_pos / en_neg` — mean EBM energy of positive/negative samples
- `kl_base` — raw KL divergence (before free-bits clamping)
- `force_reco / force_kl / force_ebm` — gradient norms at the encoder bottleneck
- `MCMC_Dynamics_E{n}` — per-step MCMC table (energy, grad norms, chain displacement)

---

## 📄 Reference

```bibtex
@article{pang2020learning,
  title={Learning Latent Space Energy-Based Prior Model},
  author={Pang, Bo and Han, Tian and Nijkamp, Erik and Zhu, Song-Chun and Wu, Ying Nian},
  journal={NeurIPS},
  year={2020}
}
```

---

## 🔒 License

This project is for research and educational purposes.
