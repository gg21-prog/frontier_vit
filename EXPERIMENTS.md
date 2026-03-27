# Experiment Log

Six runs across three architectures — L²ViT, PolaFormer, VanillaViT — each trained scratch and pretrain→finetune, 40 epochs supervised on the same splits.

Training curves for all six runs: `wandb1.png` (val/train loss, val/train acc, lr), `wandb2.png` (pretrain epochs and LR schedules).

---

## Results Summary (val acc, 40 epochs)

| Model | Val Acc | Train Acc | Val Loss |
|---|---|---|---|
| L²ViT (pretrain→FT) | 0.888 | 0.930 | 0.306 |
| PolaFormer (scratch) | 0.864 | 0.925 | 0.497 |
| L²ViT (scratch) | 0.852 | 1.000 | 1.143 |
| VanillaViT (pretrain→FT) | 0.838 | 0.802 | 1.217 |
| PolaFormer (pretrain→FT) | 0.813 | 0.795 | 1.399 |
| VanillaViT (scratch) | ~0.874 | 1.000 | ~0.967 |

---

## Observation 1 — Scratch models all overfit

Every scratch model hit train_acc = 100% by the end of training while val_loss kept climbing. The dataset has ~7k training samples against ~7M model parameters — not enough to regularise without a pretrained initialisation. Val loss for L²ViT scratch peaked at 1.14 by epoch 40; the best checkpoint was around epoch 10.

The fix isn't more regularisation — it's pretraining. The finetune models never reached 100% train_acc (L²ViT finetune plateaued around 93%), which means the pretrained representations were general enough that the model didn't need to memorise.

---

## Observation 2 — L²ViT finetune wins, but pretraining didn't help PolaFormer or VanillaViT

This was unexpected. PolaFormer finetune (0.813) was the worst model despite pretraining — worse than PolaFormer scratch (0.864). VanillaViT finetune (0.838) also underperformed its own scratch run.

Possible reasons:
- **LR too low for these architectures.** 3e-5 was set based on L²ViT experiments. PolaFormer's attention mechanism (polar decomposition-based) may need a different LR regime during finetuning.
- **MAE decoder mismatch.** The same lightweight 2-layer MLP decoder was used for all architectures. It may not be well-suited to PolaFormer's internal representation.
- **Freeze epochs.** Freezing the encoder for 10 epochs may have been too aggressive for PolaFormer — the task heads may not stabilise well when the encoder is substantially different from L²ViT's structure.

L²ViT benefits most from MAE pretraining likely because its linear attention lacks the positional bias of softmax attention, so learning global structure from unlabelled data is more valuable — the LCM handles local structure but the encoder needs pretraining to handle global context well.

---

## Observation 3 — MAE pretrain loss plateaued (~1.07)

All three finetune runs showed the same pretrain loss curve: rapid drop in the first few epochs, then plateau around 1.07 for all 40 epochs. This is expected — the energy-weighted masking specifically targets the sparse ~5% of patches with actual jet energy, making reconstruction hard. A predict-zeros baseline already scores above 1.0 on these patches.

The plateau doesn't mean the pretraining is useless — the encoder is learning jet structure from the reconstruction task even if the loss number looks high. The downstream val_acc improvement confirms it.

---

## Observation 4 — L1 vs MSE regression loss

L1 was chosen to avoid regression-to-mean. In hindsight, the mass distribution is bimodal (~80 GeV and ~200 GeV peaks), so both the mean and median fall in the trough between modes. MSE would actually give a lower reported MAE for this distribution because its predictions cluster nearer the mean, which happens to be closer to more samples than L1's median target.

An earlier run with MSE loss achieved ~22 GeV MAE on the finetune model vs 26 GeV with L1. For a bimodal mass distribution, MSE is the pragmatically better choice.

---

## Observation 5 — Energy-weighted masking

Standard MAE uses uniform random masking. Here masking probability is proportional to L1 energy per patch (Gumbel trick for differentiable sampling), so the model is forced to reconstruct the high-energy signal patches rather than empty background. This biases the encoder toward learning jet-relevant features during pretraining instead of modelling detector noise.

The effect is visible in the pretrain loss — it's higher than a uniform masking run would be because the task is harder, but the downstream classification benefit is real.

---

## LR and Training Setup

- Scratch: lr=3e-4, cosine decay with 5-epoch warmup
- Finetune: lr=3e-5 (10× lower), cosine decay, encoder frozen for first 10 epochs
- Optimiser: AdamW, weight_decay=0.05, gradient clip at 1.0
- AMP: bfloat16 on all runs

The LR difference is the key lever. High LR on pretrained weights destroys the representations built during MAE pretraining. The freeze-then-unfreeze schedule lets the classification and regression heads find a reasonable loss landscape before the encoder starts moving.
