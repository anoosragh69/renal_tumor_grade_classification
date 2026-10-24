# Task Checklist: vViT Renal Tumor Grade Classification

Authoritative plan: `tasks/plan.md` (revised 2026-10-23).
Single-owner checklist from 2026-10-23 onward: teammate split in `tasks/split.md` is **obsolete** and kept only as history.

Status legend: `[x]` done · `[~]` in progress / partial · `[ ]` pending · `[SUPERSEDED]` retained for history, not on the critical path.

---

## M0: Data, Exploration & Literature — DONE

- [x] Task 1: Literature review and paper collection
- [x] Task 2: Literature summary table
- [x] Task 3: 1-page paper summary
- [x] Task 4: Dataset download (300 cases; 210 with segmentations + `kits.json`)
- [x] Task 5: Data exploration and analysis
- [x] Step 1: Pin `requirements.txt` (commit `76dfd30`)
- [x] Step 1: Audit WHO/ISUP grade availability (commit `833b0d4`) — 172/210 labeled (low 119 / high 53), grades 1–4 only, all labeled cases have masks; GitHub mirror vs TCIA deviation documented in `docs/isup_grade_audit.md`

---

## M1: Fix cohort — real clinical fields, ~157–172 patients, new splits

**Why:** the current `clinical.csv` uses synthetic comorbidity/habit values and the current `splits.json` keeps only 66 patients (balanced + mean±1 SD applied before splitting). The revised plan uses real `kits.json` fields and keeps the cohort near the paper's 157.

- [x] Rewrite `src/data/clinical.py` from `kits.json` real fields only:
  - demographic: `age_at_nephrectomy`, `gender`, `body_mass_index`
  - comorbidity: `peripheral_vascular_disease`; DM = `uncomplicated_diabetes_mellitus` OR `diabetes_mellitus_with_end_organ_damage`; `chronic_kidney_disease`
  - habit: `smoking_history` (never / previous / current → 0/1/2), `alcohol_use` (actual 3 JSON levels → 0/1/2)
  - z-score continuous fields with **train-only** statistics (implemented in the dataset)
  - document DM definition and smoking/alcohol mapping (our assumptions; paper does not specify)
- [x] Audit missing clinical fields — **0 missing** across all 8 model-facing fields in all 210 patients; no imputation needed. Noted 4 patients with age < 18 (`case_00031`, `case_00088`, `case_00160`, `case_00162`) as a source-data flag
- [x] Retire `src/data/synthetic_clinical.py`; no `synthetic=True`/`bmi_synthetic` path remains
- [x] Update `src/data/dataset.py` to consume the real `clinical.csv`; age/BMI z-scored with **train-split** mean/std inside each dataset instance; dataset self-test passed
- [x] Fix `src/data/dataset_split.py`:
  - exclusion order per plan §3.2: missing clinical → missing ISUP → image count mean±1 SD → radiomics/mask failure
  - **no patient equalisation before splitting**
  - class balancing by image subsampling in **train only**
  - patient-level stratified split targeting 111/15/30
  - version `splits.json` + seed; produce runs for ≥3 seeds
  - add 5-fold patient-level stratified CV folds (`cv_folds.json`)
- [x] Decide and document radiomics level: per-slice 2D crop (chosen)
- [x] Re-run top-16 F-value selection **train-only, inside each fold/seed** after the new cohort/splits exist
- [x] Reconcile top-16 artifact with the dataset (2026-10-24) — `top16_features.json` had become a per-fold dict and crashed `dataset.py`; now `top16_selection.py` writes the frozen paper-split list to `top16_features.json`, per-fold lists to `top16_features_folds.json`, and regenerates `radiomics_top16.csv` to match. Loader verified: train 91 / val 12 / test 23 patients, 17,774 / 2,354 / 4,409 slices
- [ ] **Checkpoint:** cohort and split counts logged; no patient/slice leakage; `splits.json` + `cv_folds.json` committed — **open: cohort is 126, not ~157; see Discrepancies below**

---

## M2: Radiomics-only sanity gate

Run this before any neural training.

- [x] Logistic regression and random forest on the 16 radiomic features
- [x] 5-fold patient-level CV; top-16 selection fit **inside each fold**
- [x] Report patient-level AUROC mean ± SD and pooled out-of-fold AUROC (rerun 2026-10-24)
- [x] **Gate decision: FAIL.** Pooled patient-level AUROC **LR 0.534 / RF 0.580** (n=126); per-fold LR mean 0.530 ± 0.075, RF mean 0.600 ± 0.026. Accuracy is meaningless here: RF 0.762 vs **majority-class baseline 0.754** (95 low / 31 high). Both models sit in the plan's near-chance band, while the paper reports the radiomic sector as dominant — **audit before training M4**.
- [x] **Checkpoint:** gate results written to `results/metrics/radiomics_gate_*.json` and `docs/radiomics_gate_report.md`
- [x] Re-run at **patient level** (mean slice probability per patient) — done 2026-10-24, superseding the per-slice scoring

---

## Discrepancies found 2026-10-24 (audit before M3/M4)

- [ ] **Cohort is 126, not ~157.** `splits.json` keeps 91/12/23 (126 total) because `slice_count_outlier` excludes **46** patients; the paper excludes **15**. Root cause: our "image count" is the number of *tumor-containing* slices (mean 198.0, SD 34.9 over 210 patients), a very wide distribution that mean±1 SD guts. The paper almost certainly counted CT-series slices, not tumor slices. Decide: relax/replace the outlier rule, or declare 126 as our cohort in §9.
- [ ] **M2 gate near chance (LR 0.534 / RF 0.580 patient-level, n=126).** Plan §M2 says to audit labels, radiomics extraction and leakage before M4. Candidates: per-slice radiomics are noisy for a per-patient task; class imbalance (95 low / 31 high in cohort); per-tumor (3D) aggregation instead of per-slice; LR/RF may be the wrong read and the paper's own radiomics lead could depend on its 157-patient cohort.
- [ ] **Baseline runs look incomplete.** `baseline_*_test_metrics.json` all report `checkpoint_epoch: 2` and `n_slices: 320` — these are not the 200-epoch early-stopped runs the checklist describes.
- [ ] **vViT attention still does not match the paper.** `src/models/vvit.py` uses `nn.TransformerEncoder` with 8 heads / dim 128 → head_dim 16; the plan (§M4) requires inner dim 512 (8×64) projected back to 128 via a custom block. Not yet done.

---

## M3: Speed fix

Target: full vViT run in under ~1 h; a 37 h run means a bug.

- [x] Confirm `torch.cuda.is_available()` and that the active env is `cu126` — `src/data/check_gpu.py` (RTX 3050, CUDA 12.6)
- [x] Cache preprocessed crops as `.npy` — `data/processed/crops_cache/`, wired into `dataset.py`
- [ ] Time a DataLoader-only pass; time one forward/backward on one batch
- [ ] Tune `num_workers` (Windows: 2–4, `persistent_workers=True`)
- [ ] Confirm each epoch uses the balanced training crops, not all tumor slices (**note:** current loader yields all 17,774 train crops; plan expects a subsampled balanced ~1.4k)
- [ ] Confirm AMP is enabled and tensors are on GPU
- [ ] **Checkpoint:** measured epoch time + projected full-run time recorded in `LOG.md`

---

## M4: vViT — paper-style split (≥3 seeds) + 5-fold CV

- [ ] Fix attention block to the paper shape: inner dim 512 (8 heads × 64) projected back to 128; do not rely on `nn.TransformerEncoder`
- [ ] Training protocol: BCE per head averaged over batch × heads; Adam (β 0.9/0.999, ε 1e-8, wd 0, no AMSGrad), lr 1e-3; AMP; early stopping patience 15 on val accuracy; best-val checkpoint
- [ ] Image-sector-only augmentation: h-flip, v-flip, perspective, invert, posterize, solarize, equalize
- [ ] Run paper-style split for ≥3 seeds; run 5-fold patient-level CV
- [ ] Patient-level metrics primary: mean slice probability for AUROC, majority vote for labels; bootstrap CIs resampling **patients** (~1000)
- [ ] Image-level metrics secondary only, explicitly labelled as inflated by within-patient correlation
- [ ] Report accuracy, sensitivity, specificity, PPV, NPV, F-score, log loss, Cohen's κ, AUROC
- [ ] **Checkpoint:** patient-level results table + CIs for all seeds/folds; run configs saved

---

## M5: Permutation importance + figures

- [ ] Baseline accuracy of the trained model on the test split
- [ ] Per sector: shuffle patient sector values across patients (other sectors intact); recompute accuracy; repeat 100×
- [ ] Importance = drop in accuracy; pairwise Mann–Whitney U between sectors
- [ ] Boxplots + ROC figure (Fig. 4/5 equivalents)
- [ ] Compare with paper: radiomic dominant (~0.30 image-level, 0.154 patient-level); image sector near zero/negative
- [ ] **Checkpoint:** importance table + figures + comparison note

---

## M6: Optional — one image-only baseline

Only if time allows; not required for the core result.

- [x] `TimmBaseline` code (`vit_small_patch16_224` / `convnext_small` / `resnext50_32x4d`) with `in_chans=1` stem fold, 128→224 resize
- [x] Baseline non-learning diagnosis and fix (lr 1e-4, AMP, early stopping patience 15, `num_workers=2`) — `docs/baseline_training_diagnosis.md`
- [ ] Run **one** pretrained baseline (ResNet18 or ConvNeXt-tiny), lr 1e-4, paper-style split only
- [ ] DeLong / McNemar comparison only if this baseline is trained
- [ ] Do not run the full ViT/ConvNeXt/ResNeXt trio or CV on CNN baselines unless time allows
- [ ] **Checkpoint (optional):** baseline vs vViT comparison table

---

## M7: Demo

- [ ] `demo.py` on a test-split case: load CT + mask + `kits.json`, crop slices, compute radiomics with the frozen top-16 list and train z-score stats, build clinical vector, run vViT
- [ ] Print per-sector votes, final vote, and P(high grade) per slice and per patient
- [ ] State the input requirement (image crop + mask + clinical record; raw CT alone is insufficient) and present as research illustration, not diagnostic

---

## M8: Report

- [ ] Draft: Intro → Methods → Results → Discussion → Limitations
- [ ] Declare every deviation from the paper (plan §9), including:
  - Python/PyTorch versions vs 3.8/1.7.1
  - cohort/split counts vs 157 → 111/15/30
  - DM definition and alcohol-level mapping
  - radiomics per slice vs per tumor
  - loss aggregation, lr, 3–3 vote tie-break
  - early stopping + AMP vs fixed 200 epochs fp32
  - train-only class balancing; test keeps true prevalence
  - patient-level CV in addition to the single split
  - baseline pretraining status and lr
  - GitHub KiTS19 mirror vs TCIA C4KC-KiTS
- [ ] One-page README with the pipeline diagram from plan §2
- [ ] **Checkpoint:** final writeup complete

---

## Superseded work (kept for history)

- [SUPERSEDED] Old M1 pilot: 40-case mock run (8/2/3) and 210-case real run with 48/6/12 patients (9,467/1,178/2,230 slices). Replaced by the revised cohort in M1.
- [SUPERSEDED] Old M2 sanity check: 16-slice overfit (loss 0.52→0.0014); vViT integration on 48/6/12 batch — architecture still valid, data/split replaced.
- [SUPERSEDED] Old M3 training: 200 epochs on 48/6/12, best val_acc 0.9277 @ epoch 67, test image acc 0.531 / patient acc 0.500, AUROC 0.639. Severe overfit; numbers are **not** final and must not be reported as the project result. Artifacts remain in `results/` for reference.
- [SUPERSEDED] Old M4 three-baseline suite/code remains usable, but only one baseline is now in scope (M6).
- [SUPERSEDED] Old M6 attention-fusion improvement: moved out of scope (plan §11); revisit only after M1–M8.
- [SUPERSEDED] `tasks/split.md` teammate ownership — single owner from 2026-10-23.

---

## Notes

- **Artifact status:** `data/processed/clinical.csv` is now **real** (210 patients; 172 labeled; no missing model-facing fields). Current `data/processed/splits.json` is still the old 66-patient split (train 48 / val 6 / test 12), produced with pre-split equalisation and mean±1 SD; it must be regenerated next in M1. Current `top16_features.json` was selected on the old split and must be re-fitted train-only per fold/seed.
- **Dataset:** real `kits19/data/` present (210 cases with segmentation + `kits.json`; cases 00210–00299 imaging-only).
- **Radiomics env:** pyradiomics 3.1.0 in Miniconda env `radiomics` (`%USERPROFILE%\miniconda3\envs\radiomics\python.exe`; Python 3.9, numpy 1.26.4). Full run: that python + `src/data/radiomics_extraction.py --splits data/processed/splits.json --workers 7` (~50 min; resume checkpoints in `data/processed/radiomics_parts/`). `normalize=True` + binWidth=25 gives ~0.4 s/slice.
- **Training env:** torch 2.14.0+cu126, torchvision 0.29.0+cu126; `torch.cuda.is_available()=True` on RTX 3050 Laptop 6GB.
- **Docs:** dated run results and debugging history go in `LOG.md`, not here.
- **Git hygiene:** small focused commits; no data in the repo.
