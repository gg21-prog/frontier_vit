# L²ViT for Particle Jet Classification & Mass Regression

**GSoC 2026 — ML4Sci | Task 2h: Linear Attention Vision Transformers**

---

## Problem

CMS detector jet images (125×125, 8 channels) need to be classified by particle type (C0/C1) and have their mass regressed in GeV. The dataset is small (~10k labelled samples), making overfitting a real concern.

The approach: train an L²ViT encoder with masked autoencoder pretraining on 60k unlabelled jet images, then finetune on the labelled set — and compare against training from scratch.

---

## Architecture

**L²ViT** ([arXiv:2501.16182](https://arxiv.org/abs/2501.16182)) — linear-complexity attention via ReLU kernel:

```
Attn(Q,K,V) = φ(Q)(φ(K)ᵀV) / (φ(Q)Σφ(K))    φ(x) = ReLU(x)
```

O(N) vs O(N²) for softmax attention. A Local Concentration Module (LCM) adds depthwise convolutions to recover local spatial structure that linear attention loses.

Dual output head: classification (cross-entropy) + mass regression (L1).

---

## Results

| Model | Accuracy | AUC | MAE (GeV) | R² |
|---|---|---|---|---|
| L²ViT (pretrain→FT) | **0.8985** | **0.9553** | **26.05** | **0.5618** |
| L²ViT (scratch) | 0.8670 | 0.9334 | 33.18 | 0.3328 |

Pretraining improves every metric. Full analysis in `RESULTS.md`, cross-architecture comparison in `EXPERIMENTS.md`.

---

## Repo Structure

```
l2vit_particle.ipynb     Main notebook — run this
weights/                 Saved model checkpoints
RESULTS.md               L²ViT test results + training dynamics
EXPERIMENTS.md           Cross-architecture observations and analysis
```

**Notebook cell guide:**
| Cell | What it does |
|---|---|
| 1–2 | Imports, paths |
| 3 | Config — set `EXPERIMENT` and `MODE` here |
| 4 | Data loading and train/val/test split |
| 5 | EDA — jet image visualisation |
| 6 | L²ViT, PolaFormer, VanillaViT architectures |
| 7 | Training utilities |
| 9 | Run experiment (train or pretrain+finetune) |
| 11 | Evaluate all checkpoints — ROC, confusion matrix, mass distributions |
| 12 | **L²ViT focused comparison** — scratch vs finetune, Task 2h figures |

---

## Weights

| File | Description |
|---|---|
| `weights/model_l2vit_finetune.pt` | Best L²ViT finetune model (by val acc) |
| `weights/model_l2vit_scratch.pt` | Best L²ViT scratch model (by val acc) |
| `weights/encoder_l2vit_finetune.pt` | MAE pretrained encoder (40 epochs) |
| `weights/model_polaformer_*.pt` | PolaFormer checkpoints |
| `weights/model_vit_*.pt` | VanillaViT checkpoints |

---

## Reproducing

1. Place dataset files in the repo root (not included — available at the CERN CERNBox link in the task description)
2. Open `l2vit_particle.ipynb`
3. Set `EXPERIMENT` in Cell 3 to the run you want
4. Run all cells top to bottom
5. Cell 11 evaluates all available checkpoints; Cell 12 shows the L²ViT comparison

Requires: `torch`, `h5py`, `numpy`, `sklearn`, `matplotlib`, `tqdm`, `wandb`
