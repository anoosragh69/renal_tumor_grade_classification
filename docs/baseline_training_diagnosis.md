# Baseline Non-Learning Diagnosis (M4 blocker)

**Date:** 2026-10-08
**Symptom source:** full 200-epoch baseline runs, `results/metrics/baseline_*_metrics.jsonl`
**Probe:** `src/training/diagnose_baseline.py`

## Symptom

The first full timm baseline runs (Step 8) never learned:

| Observation | Value |
|---|---|
| vit train_loss after epoch ~7 | 0.6915–0.6929, flat through epoch 73 |
| Train label entropy (slice pos-rate 0.469) | 0.6931 — **loss == entropy ⇒ constant predictor** |
| vit val_acc | stuck toggling 0.4881 / 0.5119 (majority class) |
| Gradients at init (probe) | healthy: head 2.8e+01, body 3.2e+01 — not a wiring bug |

Only the head's bias learned (fitting the base rate); no discriminative signal.

## Probe design

Fixed batch of 16 real train slices, **mixed labels** (8 low / 8 high, no
augmentation), 60 Adam steps, then feature statistics. Three configs per
backbone: full finetune at lr 1e-3 (the run's default), full at 1e-4,
head-only at 1e-3.

> Early trap: a *single-class* batch (all label 0) memorises at any lr via the
> bias alone — the probe must use mixed labels to test discrimination.

## Findings

### Optimisation probe (vit, 60 steps, mixed batch)

| Config | final loss | verdict |
|---|---|---|
| lr 1e-3 full finetune | **0.6928** (ln2) | **STUCK** — logit_std collapses 0.87 → 0.007 within 10 steps |
| lr 1e-4 full finetune | 0.0001 | learns |
| lr 1e-3 head-only | 0.0269 | learns (385 params, no pretrained features to destroy) |

### Feature diversity after those 60 steps (vit)

| Config | feat_std/dim | effective rank |
|---|---|---|
| pretrained init | 1.0886 | 147/384 |
| after lr 1e-3 training | **0.0050** | **7/384 — collapsed** |
| after lr 1e-4 training | 1.7098 | 249/384 |

## Root cause

Adam at lr 1e-3 (paper specifies Adam but **not** lr; 1e-3 was our default,
matching the vViT) takes ≈±1e-3 steps per weight regardless of gradient size.
On this data the discriminative gradients are weak — inputs are low-contrast
(HU-windowed crops, per-pixel std ≈ 0.08), dataset is 48 train patients — so
destructive steps dominate, the pretrained representation collapses to a
near-constant feature vector, and only the bias survives → loss pins at label
entropy. The from-scratch vViT has no pretrained features to destroy and its
own initial scale suits 1e-3, so it trains fine (hence the bug only surfaced
in M4).

## Fix (applied 2026-10-08)

`src/training/run_baselines.py`:
1. **lr default 1e-4** for baselines (vViT keeps 1e-3); both documented in §6 deviations.
2. **AMP** — `torch.autocast` + `GradScaler` (`--no-amp` to disable).
3. **Early stopping** — patience 15 on val accuracy (`--patience`, 0=off); 200-epoch budget retained as the cap.
4. **num_workers default 2** — PIL augmentations were serialised with GPU at 0.

`src/training/run_training.py` (vViT): AMP + early stopping flags added, same
defaults, for protocol parity on the re-run.

### Verification (real kits19, full protocol)

| Run | train_loss | val_acc | epoch time |
|---|---|---|---|
| old vit (lr 1e-3), epochs 7–73 | pinned 0.692 | stuck 0.5119 | ~100 s |
| new vit (lr 1e-4 + AMP), epoch 1→3 | 0.676 → **0.533** | 0.63 (ep 2) | **57 s** |

Early-stop path verified (`--patience 1` stopped exactly on first non-improving
epoch); convnext + resnext smoke-tested end-to-end (train → checkpoint → test
metrics) under AMP.

## Consequences for results on disk

- `results/metrics/baseline_*_test_metrics.json` were **smoke artifacts**
  (2-epoch checkpoints, eval capped at 10 batches / 2 patients) — not valid
  Table 3 numbers. Superseded when full runs complete.
- Full runs required: `python src/training/run_baselines.py`
- vViT re-run recommended under the same early-stop/AMP protocol for like-for-like
  Table 3: `python src/training/run_training.py` then `run_evaluation.py` + `metrics_table.py`.

## Reproduce

```
python src/training/diagnose_baseline.py --mixed-batch --models vit --steps 60
python src/training/diagnose_baseline.py --feature-probe --models vit --steps 60 --mixed-batch
```
