# L²ViT for Particle Jet Classification & Mass Regression

**GSoC 2026 — ML4Sci | Task 2h: Linear Attention Vision Transformers**

---

## Problem

CMS detector jet images (125×125, 8 channels) need to be classified by particle type (C0/C1) and have their mass regressed in GeV. The dataset is small (~10k labelled samples), making overfitting a real concern with a from-scratch model.

---

## What's in this repo

**Task 2h deliverables** — L²ViT trained from scratch and pretrain→finetune, test results, model weights.

**Additional experiments** — the same training pipeline was also run on PolaFormer and VanillaViT (softmax attention baseline) for comparison. All three architectures share the same notebook, training utilities, and evaluation cell.

---

## Architectures

**L²ViT** ([arXiv:2501.16182](https://arxiv.org/abs/2501.16182)) — replaces O(N²) softmax attention with O(N) ReLU-kernel linear attention. A Local Concentration Module (LCM) adds depthwise convolutions to recover local spatial structure. This is the primary model for Task 2h.

**PolaFormer** — polar-decomposition-based linear attention variant, included as an alternative linear attention approach.

**VanillaViT** — standard softmax self-attention, included as a baseline.

All three use a dual output head: classification (cross-entropy) + mass regression (L1).

---

## Training regimes

**Scratch** — supervised training from random initialisation on the labelled dataset.

**Pretrain→Finetune** — 40 epochs masked autoencoder (MAE) pretraining on 60k unlabelled jet images, followed by 40 epochs supervised finetuning at 10× lower learning rate with encoder frozen for the first 10 epochs.

---

## L²ViT Test Results (Task 2h)

Evaluated on the held-out 20% test set, never seen during training.

| Model | Accuracy | AUC | MAE (GeV) | R² |
|---|---|---|---|---|
| L²ViT (pretrain→FT) | **0.8985** | **0.9553** | **26.05** | **0.5618** |
| L²ViT (scratch) | 0.8670 | 0.9334 | 33.18 | 0.3328 |

Full analysis in `RESULTS.md`. Cross-architecture observations in `EXPERIMENTS.md`.

---

## Repo structure

```
l2vit_particle.ipynb        Single notebook — all architectures, training, and eval
weights/                    Saved model checkpoints (all 6 runs)
figures/
  task_2h/                  L²ViT scratch vs finetune comparison (Task 2h)
    roc_l2vit_comparison.png
    cm_l2vit_comparison.png
    mass_l2vit_comparison.png
  experiments/              All-model eval plots and training curves
    roc_curves.png
    confusion_matrices.png
    mass_distributions.png
    jet_eda.png
    wandb1.png              WandB training curves — all 6 runs
    wandb2.png              WandB pretrain epochs and LR schedules
RESULTS.md                  L²ViT test results, training dynamics, regression analysis
EXPERIMENTS.md              Cross-architecture observations from all 6 runs
```

---

## Notebook guide

| Cell | What it does |
|---|---|
| 3 | Config — set `EXPERIMENT` here before running |
| 4 | Data loading and stratified train/val/test split |
| 5 | EDA — jet image visualisation per class |
| 6 | L²ViT, PolaFormer, VanillaViT architecture definitions |
| 9 | Run experiment (handles scratch or pretrain→finetune depending on `EXPERIMENT`) |
| 11 | Evaluate all available checkpoints — ROC, confusion matrix, mass distributions |
| 12 | **Task 2h focused cell** — L²ViT scratch vs finetune comparison only |

---

## Weights

| File | Description |
|---|---|
| `weights/model_l2vit_finetune.pt` | L²ViT pretrain→finetune (best val acc) |
| `weights/model_l2vit_scratch.pt` | L²ViT scratch (best val acc) |
| `weights/encoder_l2vit_finetune.pt` | MAE pretrained L²ViT encoder |
| `weights/model_polaformer_finetune.pt` | PolaFormer pretrain→finetune |
| `weights/model_polaformer_scratch.pt` | PolaFormer scratch |
| `weights/model_vit_finetune.pt` | VanillaViT pretrain→finetune |
| `weights/model_vit_scratch.pt` | VanillaViT scratch |

---

## Reproducing

1. Place `Dataset_Specific_Unlabelled.h5` and `Dataset_Specific_labelled_full_only_for_2i.h5` in the repo root (not included — available via the CERN CERNBox link in the task description)
2. Open `l2vit_particle.ipynb` and set `EXPERIMENT` in Cell 3
3. Run all cells top to bottom
4. Cell 11 evaluates all checkpoints in `weights/`; Cell 12 shows the Task 2h L²ViT comparison

Dependencies: `torch`, `h5py`, `numpy`, `sklearn`, `matplotlib`, `tqdm`, `wandb`
