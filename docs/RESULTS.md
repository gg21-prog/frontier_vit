# L²ViT Results — Task 2h

**GSoC 2026 · ML4Sci | Linear Attention Vision Transformer for Particle Mass Regression & Classification**

Hardware: NVIDIA RTX 4060 Laptop GPU

---

## Setup

- **Labelled dataset:** `Dataset_Specific_labelled_full_only_for_2i.h5` (~10k samples)
- **Unlabelled dataset:** `Dataset_Specific_Unlabelled.h5` (60k samples, MAE pretraining only)
- **Split:** 70% train / 10% val / 20% test, stratified by class — test set never touched during training
- **Task:** Binary classification (C0 vs C1) + continuous mass regression

---

## Architecture

L²ViT ([arXiv:2501.16182](https://arxiv.org/abs/2501.16182)) replaces O(N²) softmax attention with O(N) ReLU-kernel linear attention:

```
Attn(Q,K,V) = φ(Q)(φ(K)ᵀV) / (φ(Q)Σφ(K))   where φ(x) = ReLU(x)
```

A **Local Concentration Module (LCM)** — two 7×7 depthwise convolutions — re-introduces local spatial structure on the patch grid.

| | Value |
|---|---|
| Input | 125×125, 8 channels |
| Patch size | 5×5 → 625 tokens |
| Embed dim | 192 |
| Depth / Heads | 6 / 4 |
| Parameters | ~7M |

Dual head: shared encoder → classification (cross-entropy) + regression (L1, normalised mass). Lambda auto-calibrated before each run.

---

## Experiments

Both experiments trained for **40 supervised epochs** on the same data splits.

**Scratch** — random initialisation, lr=3e-4, cosine decay.

**Pretrain→Finetune** — 40 epochs MAE pretraining on unlabelled data (mask ratio 0.75, energy-weighted masking), then 40 epochs supervised with lr=3e-5. Encoder frozen for first 10 epochs of finetuning.

---

## L²ViT Test Results

All metrics on the **held-out 20% test set**.

| Model | Accuracy | AUC | MAE (GeV) | R² |
|---|---|---|---|---|
| L²ViT (scratch) | 0.8670 | 0.9334 | 33.18 | 0.3328 |
| L²ViT (pretrain→FT) | **0.8985** | **0.9553** | **26.05** | **0.5618** |
| Δ | +3.15% | +0.022 | −21.5% | +0.229 |

Pretraining improves every metric. Figures: `../figures/task_2h/roc_l2vit_comparison.png`, `../figures/task_2h/cm_l2vit_comparison.png`, `../figures/task_2h/mass_l2vit_comparison.png`.

---

## Training Dynamics

**Scratch** — val loss hit minimum (~0.38) at epoch 6, then diverged to 1.14 by epoch 40. Train accuracy reached 100% by epoch 28. Classic overfitting: ~7M parameters on ~7k training samples with no pretrained initialisation. Model saved at best val_acc checkpoint (~epoch 10).

**Finetune** — val loss and val acc both converged cleanly. Encoder freeze for the first 10 epochs let the task heads warm up before gradients flowed back through the encoder. Final val_acc ≈ 0.888.

---

## Classification

- Scratch AUC **0.9334** vs finetune **0.9553** (+0.022)
- Finetune produces a more balanced confusion matrix — pretraining on unlabelled data forces the encoder to learn general jet structure before class labels are introduced
- Both plots in `../figures/task_2h/cm_l2vit_comparison.png`, ROC in `../figures/task_2h/roc_l2vit_comparison.png`

---

## Regression

- Scratch MAE **33.18 GeV**, R² **0.3328** — regression head overfit; limited predictive power
- Finetune MAE **26.05 GeV** (−21.5%), R² **0.5618** — meaningfully better

The mass distribution is bimodal (~80 GeV and ~200 GeV). L1 loss targets the median, which falls in the valley between modes — neither class is well-predicted at that point. For this dataset, MSE loss would give lower reported MAE. The switch to L1 was reasonable in principle but hurt the regression metric here. Mass distribution plots in `../figures/task_2h/mass_l2vit_comparison.png`.

---

## Overall Comparison

All three architectures, same 40-epoch budget.

| Model | Val Acc | Notes |
|---|---|---|
| L²ViT (pretrain→FT) | **0.888** | Best overall |
| PolaFormer (scratch) | 0.864 | Better than its own finetune run |
| L²ViT (scratch) | 0.852 | Overfits |
| VanillaViT (pretrain→FT) | 0.838 | |
| PolaFormer (pretrain→FT) | 0.813 | Pretraining hurt — see EXPERIMENTS.md |
| VanillaViT (scratch) | ~0.874 | Overfits |

Full per-model plots produced by Cell 11 in the notebook.
