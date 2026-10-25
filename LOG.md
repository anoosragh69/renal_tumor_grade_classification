# LOG — dated run results and debugging history

Plan: `tasks/plan.md`. Checklist: `tasks/todo.md`. Only dated, factual notes live here.

---

## 2026-10-25

### vViT multi-run preparation

- Added split-specific input selection to the vViT training and evaluation scripts, stable run tags for separate checkpoint/metric artifacts, and split-specific train-only top-16 feature selection. Train crop balancing is seeded from each run's seed.
- Added train-only z-scaling for radiomic inputs. Training-patient rows determine the means and standard deviations; each checkpoint records those values with its input artifact paths. The earlier two-epoch vViT probe predates this scaling and is not a final result.
- Generated separate top-16 feature/radiomics artifacts for paper seeds 43 and 44.
- Added and ran `build_nested_cv_splits.py`. Each outer fold remains untouched for test evaluation; an inner stratified 15% patient subset of outer training is used for early stopping. Five split and feature-selection artifact sets are in ignored `data/processed/` paths. Unlabeled patients in `clinical.csv` are excluded when mapping CV labels.
- At the time of this preparation note, paper-style and CV training/evaluation were pending; those runs are recorded below.

### vViT paper-split and CV runs

- Completed paper-style seeds 42–44 and all five nested CV folds with the configured protocol. Early stopping ran 21, 22, 30, 20, 27, 21, 32, and 20 epochs respectively; best epochs ranged from 5 to 17. Each checkpoint was evaluated once on its held-out patients.
- Paper-split patient AUROCs for seeds 42/43/44: **0.611 / 0.646 / 0.540** (test n=31 per split; wide CIs). The test cohorts overlap across split seeds, so treat these as split variability rather than independent replications.
- CV patient AUROC mean ± SD across folds: **0.618 ± 0.047**; patient accuracy **0.619 ± 0.095**. Pooled out-of-fold results across 171 patients: accuracy **0.620**, AUROC **0.608** (95% patient-bootstrap CI 0.510–0.699).
- Full seed/fold table and interpretation: `docs/vvit_results_2026-10-25.md`. No performance tuning was done.

### Image baseline suite on the 171-patient cohort

- Completed ViT, ConvNeXt, and ResNeXt training on the seed-42 split using the 200-epoch cap, patience 15, lr 1e-4, AMP, and pretrained weights. Best validation-accuracy checkpoints were at epochs 4, 3, and 3; runs stopped after 19, 18, and 18 epochs respectively.
- Evaluated all checkpoints on the untouched 31-patient / 6,348-slice test split. Patient AUROC: ViT **0.732** (95% CI 0.489–0.958), ConvNeXt **0.657** (0.423–0.881), ResNeXt **0.722** (0.484–0.929). Patient accuracy: **0.806**, **0.710**, and **0.645** respectively. Image AUROC: **0.637**, **0.583**, and **0.609**.
- The patient-level intervals are wide at n=31. Results are descriptive; no tuning for improved accuracy is planned. Detailed table and interpretation: `docs/baseline_results_2026-10-25.md`.

## 2026-10-24

### Cohort fix + M1/M2/M3 re-run on the 171-patient cohort (supersedes the 126-cohort numbers below)

- **Image-count rule changed (MD-1).** `dataset_split.py` rewritten: exclusion order is
  missing clinical → missing ISUP → **image count mean±1 SD** → radiomics failure → no tumour
  slice with mask ≥ 256 px; no pre-split equalisation. "Image count" is now the **CT-series
  slice count** (from `imaging.nii.gz`), the literal reading of images per patient.
  - No reading reproduces the paper's 15 exclusions: CT-series excludes **1** (`case_00160`),
    tumour-slice excludes 46, log-space 46, robust 4. Chose CT-series → 171 patients.
  - **Cohort: 171 patients (119 low / 52 high).** Exclusions: 38 no ISUP grade,
    1 image-count outlier, 0 radiomics failure, 0 no-valid-slice.
  - Splits seed 42: train 124 (86 low / 38 high), val 16 (11/5), test 31 (22/9).
    Also written: `splits_seed43.json`, `splits_seed44.json`, `cv_folds.json` (5-fold, 34–35 val each).
- **Train-only class balancing (plan §3.3).** `dataset.py` subsamples train crops:
  24,200 → **15,936 (7,968/class)**; val (3,101) and test (6,348) keep true prevalence.
  Dataset self-test passed.
- **M2 gate re-run, patient level, n=171 (MD-2).** Pooled AUROC **LR 0.626 (95% CI 0.533–0.705),
  RF 0.636 (0.548–0.720)**; RF accuracy 0.702 vs majority baseline 0.696.
  Patient-aggregated variant (features averaged per patient before training) **LR 0.599 /
  RF 0.597** — so per-slice noise is not the cause; radiomics is genuinely weak here.
  Verdict: **weak-signal pass with caveats** (CIs exclude 0.5, but far below the paper's
  dominant-radiomics claim). Report regenerated: `docs/radiomics_gate_report.md`.
- **M3 timing re-measured** on the balanced 15,936-crop train set: epoch 1 **34.6 s** (warm
  cache), steady state **14.2 s/epoch** → 200 epochs ≈ **47 min** (plan §6 "≲1 h" met).

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

### Cohort observation (open, analysis added 2026-10-24)

`splits.json` keeps 126 patients because `slice_count_outlier` excludes 46, where the paper
excludes 15. Three readings of "image count" were tested on the 172 labeled patients:

| Interpretation | Excluded | Kept |
|---|---|---|
| Tumor-containing slices (mask > 0, current) | 46 | 126 |
| Tumor-containing slices with mask >= 256 px | 46 | 126 |
| CT-series slices (`imaging.nii.gz` z size, mean 513.7 +/- 21.6) | 1 (`case_00160`) | 171 |

Band detail for the tumor-slice reading: mean 175.9 +/- 34.5, 22 patients above the upper
bound and 24 below the lower; log-space 1 SD also excludes 46; median +/- 3 robust SD
excludes 4. **No reading reproduces the paper's 15**, so the choice is declarative, not a
recovery of the paper's exact rule. Counts are genuine (tumors span ~166-216 axial slices;
verified against `segmentation.nii.gz` for sampled cases). Unresolved — pending decision.

### vViT attention block replaced (M4 step 1)

- `nn.TransformerEncoder` cannot express heads 8 x head_dim 64 with embed 128 (it ties
  head_dim = d_model / nheads = 16), so this had been a declared deviation.
- New `PaperAttention`: q/k/v 128 -> 512, attention with 8 heads x 64, projection 512 -> 128;
  `EncoderBlock` keeps the pre-norm residual form and GELU MLP of dim 32.
- Self-test passes; parameters 4.27M; steady-state epoch 35.2 s (was 32.8 s); epoch-1
  train_loss 0.2416 (was 0.2604), deterministic across repeat runs.
- Checkpoints from the timing run were superseded by this one.

