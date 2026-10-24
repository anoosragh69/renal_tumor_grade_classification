# LOG — dated run results and debugging history

Plan: `tasks/plan.md`. Checklist: `tasks/todo.md`. Only dated, factual notes live here.

---

## 2026-10-24

### M3 speed measurement (single full epoch, RTX 3050, AMP, num_workers=2)

| Measurement | Value |
|---|---|
| First epoch (cold crop cache) | **167.3 s** — one-time cost while `data/processed/crops_cache/` fills |
| Steady-state epoch (warm cache) | **35.7 s** (train 17,774 slices + val 2,354 slices); **32.8 s** after `persistent_workers=True` (workers no longer respawn each epoch) |
| Data loading + augmentation | 1,169 slices/s with `num_workers=2` + `persistent_workers`; 1,087 at `num_workers=2` without it; 417 at `num_workers=0` |
| Forward + backward | 32 ms/batch (batch 32, AMP, CUDA) |
| Validation pass | 1.4 s |
| Accounting | 16 s load + 18 s fwd/bwd + 1.4 s val ≈ 35 s ≈ measured epoch |

**Projection:** 32.8 s/epoch → 100 epochs ≈ 55 min, 200 epochs ≈ 1.8 h (plan §6 target "≲1 h"
is reachable if early stopping fires, as patience-15 runs in the old protocol stopped gaining
long before 200). The old 37 h estimate was cache warmup and `num_workers=0`, not data size.

Environment check: `device: cuda`, AMP enabled, `torch.cuda.is_available()` True (torch 2.14.0+cu126).

Superseded checkpoints from the old 48/6/12 split were renamed to
`vvit_best_superseded_48-6-12.pt` / `vvit_last_superseded_48-6-12.pt` before the timing run.

### M2 radiomics-only gate rerun at patient level

- Script: `src/data/radiomics_gate.py` (patient aggregation added 2026-10-24; LR now gets
  fold-local standard scaling — lbfgs was not converging on raw radiomic features).
- Pooled patient-level AUROC over 126 patients: **LR 0.534, RF 0.580**.
- Per-fold: LR mean 0.530 ± 0.075 (0.439–0.640), RF mean 0.600 ± 0.026 (0.561–0.632).
- Accuracy: LR 0.571, RF 0.762 vs **majority-class baseline 0.754** (95 low / 31 high).
- Gate verdict: **FAIL** (near-chance band). Report: `docs/radiomics_gate_report.md`.
- Implication: the paper's radiomics-dominant claim is not reproduced on this cohort. Audit
  before M4 training (see `tasks/todo.md` → Discrepancies).

### Top-16 selection reconciliation (blocker removed)

- `top16_features.json` had become a per-fold dict, which crashed `dataset.py`
  (`TypeError: Passing a dict as an indexer is not supported`).
- `top16_selection.py` now writes: frozen paper-split list → `top16_features.json`,
  per-fold lists → `top16_features_folds.json`, and regenerates `radiomics_top16.csv`.
- New frozen top-16 (train-only, `splits.json` train patients): firstorder Minimum,
  Uniformity, Variance; glcm Autocorrelation, ClusterTendency, JointAverage, JointEnergy,
  MaximumProbability, SumAverage, SumSquares; gldm GrayLevelVariance, HighGrayLevelEmphasis,
  LargeDependenceHighGrayLevelEmphasis, LowGrayLevelEmphasis; ngtdm Busyness, Contrast.
- Loader verified on the 126-patient split: train 91 / val 12 / test 23 patients,
  17,774 / 2,354 / 4,409 slices.

### Cohort observation (open)

`splits.json` keeps 126 patients because `slice_count_outlier` excludes 46, where the paper
excludes 15. Our slice count is *tumor-containing* slices (mean 198.0 ± 34.9 over 210
patients), not CT-series slices, so mean ± 1 SD is far more aggressive. Unresolved — decide
before presenting cohort numbers.
