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
- [x] **Checkpoint:** cohort and split counts logged; no patient/slice leakage; `splits.json` + `cv_folds.json` committed — **2026-10-24:** cohort **171** patients (119 low / 52 high); splits seed 42 train 124 (86/38) / val 16 (11/5) / test 31 (22/9); train balanced 24,200 → 15,936 crops (7,968/class); val/test left unbalanced; CV folds 34–35 val each. Deviation declared: image-count rule uses CT-series slices (see MD-1)

---

## M2: Radiomics-only sanity gate

Run this before any neural training.

- [x] Logistic regression and random forest on the 16 radiomic features
- [x] 5-fold patient-level CV; top-16 selection fit **inside each fold**
- [x] Report patient-level AUROC mean ± SD and pooled out-of-fold AUROC (rerun 2026-10-24)
- [x] **Gate decision (rerun 2026-10-24 on the 171 cohort): WEAK-SIGNAL PASS with caveats.** Pooled patient-level AUROC **LR 0.626 (95% CI 0.533–0.705) / RF 0.636 (0.548–0.720)**, n=171. CIs exclude 0.5, so signal is real but weak; RF accuracy 0.702 vs **majority-class baseline 0.696** (119 low / 52 high) is at baseline. Patient-aggregated variant (features averaged per patient before training) LR 0.599 / RF 0.597 — aggregation does **not** recover the paper's dominant-radiomics claim, so the weakness is genuine, not per-slice noise. Supersedes the 126-cohort result (LR 0.534 / RF 0.580). Audit performed; proceed to M4 with the caveat recorded.
- [x] **Checkpoint:** gate results written to `results/metrics/radiomics_gate_*.json` and `docs/radiomics_gate_report.md`
- [x] Re-run at **patient level** (mean slice probability per patient) — done 2026-10-24, superseding the per-slice scoring

---

## Discrepancies found 2026-10-24 (audit before M3/M4)

- [x] **MD-1 Cohort was 126, not ~157 — RESOLVED 2026-10-24.** `splits.json` used tumor-slice counts, excluding **46** patients; the paper excludes **15**. No reading reproduces 15: CT-series counts exclude **1** (only `case_00160`), tumor-slice counts 46, log-space 46, robust 4. **Decision (declared deviation):** use **CT-series** slice counts (literal "images per patient"), giving a **171**-patient cohort inside the plan's 157–172 target. No equalisation. See `LOG.md` 2026-10-24.
- [x] **MD-2 M2 gate near chance — RESOLVED (audited) 2026-10-24.** Re-run at patient level on the 171 cohort: LR 0.626 / RF 0.636, CIs exclude 0.5 (weak but real signal, still far below the paper's radiomics dominance). Patient-aggregated variant ~0.60 rules out per-slice noise as the cause. Labels, extraction and leakage audited clean (patient-level splits disjoint; radiomics recomputed). Proceeding to M4 with the caveat that radiomics alone is weak in this cohort.
- [x] **Baseline suite completed on the 171-patient cohort (2026-10-25).** ViT, ConvNeXt, and ResNeXt all trained with the configured 200-epoch cap / patience 15 and evaluated on 31 test patients (6,348 slices). Best validation accuracy was at epochs 3–4. Patient AUROC: ViT 0.732 (95% CI 0.489–0.958), ConvNeXt 0.657 (0.423–0.881), ResNeXt 0.722 (0.484–0.929). These are indicative estimates with wide intervals; no accuracy-tuning follow-up is planned. Results: `docs/baseline_results_2026-10-25.md`.
- [x] **vViT attention mismatch — RESOLVED 2026-10-24 (`8983495`).** `src/models/vvit.py` now uses a custom `PaperAttention` + `EncoderBlock`: q/k/v project 128 → 8 heads × head_dim 64 (inner 512), attention at head_dim 64, output projected 512 → 128. Self-test asserts `inner_dim == 512`; model is 4.27M params and deterministic across runs.

---

## M3: Speed fix

Target: full vViT run in under ~1 h; a 37 h run means a bug.

- [x] Confirm `torch.cuda.is_available()` and that the active env is `cu126` — `src/data/check_gpu.py` (RTX 3050, CUDA 12.6)
- [x] Cache preprocessed crops as `.npy` — `data/processed/crops_cache/`, wired into `dataset.py`
- [x] Time a DataLoader-only pass; time one forward/backward on one batch — **2026-10-24:** steady-state epoch **35.7 s** (16 s load + 18 s fwd/bwd at 32 ms/batch + 1.4 s val); cold cache 167.3 s one-time. See `LOG.md`
- [x] Tune `num_workers` (Windows) — `num_workers=2` + `persistent_workers=True`: 1,169 slices/s vs 417 slices/s at `num_workers=0`
- [x] Confirm each epoch uses the balanced training crops, not all tumor slices — 2026-10-24: train loader subsamples to **15,936 crops (7,968/class)** from 24,200; val/test unbalanced
- [x] Confirm AMP is enabled and tensors are on GPU — run reports `device: cuda`, `| amp`
- [x] **Checkpoint:** measured epoch time + projected full-run time recorded in `LOG.md` — 2026-10-24 on the balanced 15,936-crop train set: epoch 1 **34.6 s** (warm), steady state **14.2 s/epoch** → 200 epochs ≈ **47 min**; earlier 32.8–35.2 s figures were on the 24,200-crop unbalanced set. Plan §6 "≲1 h" target met

---

## M4: vViT — paper-style split (≥3 seeds) + 5-fold CV

- [x] Fix attention block to the paper shape: inner dim 512 (8 heads × 64) projected back to 128 — done 2026-10-24, `PaperAttention` + `EncoderBlock` in `src/models/vvit.py` (self-test asserts shape; 4.27M params; epoch 32.8 → 35.2 s)
- [x] Training protocol implemented: BCE per head averaged over batch × heads; Adam (β 0.9/0.999, ε 1e-8, wd 0, no AMSGrad), lr 1e-3; AMP; early stopping patience 15 on val accuracy; best-val checkpoint
- [x] Image-sector-only augmentation implemented: h-flip, v-flip, perspective, invert, posterize, solarize, equalize
- [x] Train-only radiomic z-scaling added. Means/stds are fit from training-patient rows, applied to the requested split, and recorded with each checkpoint.
- [x] Run isolation added: training/evaluation accept split, top-16, and radiomics artifact paths; unique `--run-name` values keep checkpoints, epoch logs, summaries, metrics, and predictions separate. Top-16 selection accepts a split file and output suffix, so each paper-style seed uses train-only feature selection. Train crop balancing also uses the run seed; evaluation rejects checkpoints without saved scaler metadata.
- [x] Generate train-only top-16 features and radiomics inputs for the seed-43 and seed-44 paper splits.
- [x] Run paper-style split for ≥3 seeds. Seeds 42–44 completed; full results in `docs/vvit_results_2026-10-25.md`.
- [x] Add `src/data/build_nested_cv_splits.py` and generate five inner train/validation plus untouched outer-test artifacts from `cv_folds.json`; the fold files have 115–116 train, 21 validation, and 34–35 outer-test patients.
- [x] Select train-only top-16 features and radiomics inputs separately for all five nested CV folds.
- [x] Train/evaluate all five CV folds using nested split artifacts and matching train-only top-16/radiomics files; pooled out-of-fold metrics include all 171 patients.
- [x] Patient-level evaluation supports mean slice probability for AUROC, majority vote for labels, and bootstrap CIs resampling patients (~1000)
- [x] Image-level metrics are reported as secondary and labelled as inflated by within-patient correlation
- [x] Evaluation supports accuracy, sensitivity, specificity, PPV, NPV, F-score, log loss, Cohen's κ, and AUROC
- [x] **Checkpoint:** patient-level results table + CIs for all seeds/folds; tagged run configs saved in `results/metrics/` and checkpoints.

---

## M5: Permutation importance + figures

- [x] Baseline accuracy of the seed-42 vViT on the test split is reported in the seed results artifacts.
- [x] Per sector: shuffle patient sector values across patients (other sectors intact); recompute accuracy; repeat 100×. Slice profiles are aligned by rank when patients have different slice counts.
- [x] Importance = drop in accuracy; pairwise Mann–Whitney U between sectors (exploratory; repeats reuse the same test set and p-values are unadjusted)
- [x] Boxplots + image/patient ROC figures saved under `results/figures/` (generated files are git-ignored)
- [x] Compare with paper: this seed-42 run does not reproduce radiomics-dominant importance; clinical-sector permutations produced larger accuracy drops. See `docs/vvit_permutation_importance_2026-10-25.md`.
- [x] **Checkpoint:** importance table + figures + comparison note; outputs summarized in `docs/vvit_permutation_importance_2026-10-25.md`

---

## M6: Optional — one image-only baseline

Only if time allows; not required for the core result.

- [x] `TimmBaseline` code (`vit_small_patch16_224` / `convnext_small` / `resnext50_32x4d`) with `in_chans=1` stem fold, 128→224 resize
- [x] Baseline non-learning diagnosis and fix (lr 1e-4, AMP, early stopping patience 15, `num_workers=2`) — `docs/baseline_training_diagnosis.md`
- [x] Run pretrained image baselines — the full ViT, ConvNeXt, and ResNeXt suite completed on the paper-style seed-42 split (2026-10-25; broader than the planned single baseline)
- [x] DeLong / exact McNemar comparison of vViT and image baselines on the shared patient-level test set; exploratory low-power results documented in `docs/vvit_baseline_comparison_2026-10-25.md`
- [x] No additional baseline training planned; the full ViT/ConvNeXt/ResNeXt suite has already run. No CNN baseline CV planned.
- [x] **Checkpoint (optional):** baseline vs vViT comparison table

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

- **Artifact status (2026-10-25):** paper seeds 42–44 and all five nested CV training/evaluation runs are complete; results are summarized in `docs/vvit_results_2026-10-25.md`. Seed/fold checkpoints, predictions, and scaler metadata are in ignored `results/`. The earlier two-epoch vViT probe predates radiomic scaling and is not a final result. Permutation importance and final report remain. Baseline metrics and predictions are present under `results/metrics/`.
- **Dataset:** real `kits19/data/` present (210 cases with segmentation + `kits.json`; cases 00210–00299 imaging-only).
- **Radiomics env:** pyradiomics 3.1.0 in Miniconda env `radiomics` (`%USERPROFILE%\miniconda3\envs\radiomics\python.exe`; Python 3.9, numpy 1.26.4). Full run: that python + `src/data/radiomics_extraction.py --splits data/processed/splits.json --workers 7` (~50 min; resume checkpoints in `data/processed/radiomics_parts/`). `normalize=True` + binWidth=25 gives ~0.4 s/slice.
- **Training env:** torch 2.14.0+cu126, torchvision 0.29.0+cu126; `torch.cuda.is_available()=True` on RTX 3050 Laptop 6GB.
- **Docs:** dated run results and debugging history go in `LOG.md`, not here.
- **Git hygiene:** small focused commits; no data in the repo.
