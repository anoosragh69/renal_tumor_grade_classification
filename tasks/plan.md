# Plan: vViT Renal Tumor Grade Classification (Revised 2026-10-23)

**Source paper:** Usuzaki et al., *Multimodal deep learning for predicting WHO/ISUP grading in renal tumors on CT using a self-attention-based model: variable Vision Transformer (vViT)*, Eur J Radiol Open 17 (2026) 100796

> This file is the **plan only** (what and why). Dated status notes, run results and debugging history go in `LOG.md`.

---

## 0. Goal and success criteria

**Goal:** reproduce the *method* of the paper on the same public data, evaluate it honestly, and understand what drives the prediction. The goal is **not** to match the paper's exact numbers.

**Why numbers are not the target:** the paper's test set is 30 patients (7 high-grade). Its patient-level AUROC is 0.795 with a 95% CI of 0.59–0.96, and its image-level metrics are inflated because slices from one patient are correlated.

**Project succeeds if:**
1. The pipeline is clean, leak-free and documented.
2. Patient-level results are reported with patient-level confidence intervals.
3. Permutation importance is computed and compared with the paper's finding (radiomic sector dominant; image sector near zero or negative).
4. Every deviation from the paper is declared.

---

## 1. Key decisions (locked)

| Topic | Decision |
|---|---|
| Data | KiTS19 public cases (case_00000–00209) = same 210 patients as the paper's C4KC-KiTS set. ~172 have a WHO/ISUP grade. |
| Clinical fields | **Real**, from `kits.json`. **No synthetic data.** (Earlier plan wrongly assumed these were unavailable.) |
| Task | Binary: low = WHO/ISUP 1–2, high = 3–4 |
| Model | vViT as in the paper (6 sectors, voting head) |
| Evaluation | **Patient-level is primary.** Image-level is secondary and flagged as inflated. |
| Validation | Paper-style 111/15/30 split (several seeds) **and** 5-fold patient-level stratified CV for cheap models |
| Compute | Dataset is tiny (~1.4k crops at 128×128); long runs mean a bug, not a data-size problem (see §6) |

---

## 2. Pipeline at a glance

```
KiTS19 CT + tumor mask + kits.json
        |
        +--> per-slice min-rectangle crop of tumor --> 128x128 (LANCZOS) --> image sector (16,384)
        |                                         \
        |                                          +--> PyRadiomics (105 features) --> top-16 by F-value
        |                                                (train only) --> z-score (train stats) --> radiomic sector (16)
        +--> kits.json --> demographic (3) | comorbidity (3) | habit (2)

Tokens: [class, demographic, comorbidity, habit, radiomic, image]  -> (batch, 6, 128)
   -> transformer encoder (depth 8) -> 6 output tokens -> 6 linear heads -> majority vote
   -> slice-level prediction -> aggregate per patient -> patient-level metrics
```

---

## 3. Data

### 3.1 Clinical field mapping (from `kits.json`)

| Sector | Paper field | `kits.json` source | Notes |
|---|---|---|---|
| Demographic (3) | age, sex, BMI | `age_at_nephrectomy`, `gender`, `body_mass_index` | z-score continuous values with train stats |
| Comorbidity (3) | PVD, DM, CKD | `comorbidities.peripheral_vascular_disease`; DM = `uncomplicated_diabetes_mellitus` OR `diabetes_mellitus_with_end_organ_damage`; `comorbidities.chronic_kidney_disease` | DM definition is **our assumption**; paper does not specify |
| Habit (2) | smoking, alcohol | `smoking_history` (never / previous / current); `alcohol_use` | Actual JSON has **3** alcohol levels (`never_or_not_in_last_3mo` / `two_or_less_daily` / `more_than_two_daily`), not 4. Encoded 0/1/2, as are smoking levels. Mapping is our choice; documented in `src/data/clinical.py` |

Check for nulls. The paper reports no exclusions for missing clinical data. If you find any, impute with the train median or mode and log it.

### 3.2 Exclusions (follow paper Fig. 2)
1. Missing clinical fields (expected ~0).
2. Missing WHO/ISUP grade (expected ~38 patients, leaving ~172).
3. Image count beyond mean ± 1 SD (paper: 15 patients, leaving 157: 109 low / 48 high).
4. Radiomics extraction failure or tumor mask < 256 pixels (paper: 0).
5. Class balancing by image subsampling. Apply to **training data only** (see §3.3).

Log actual counts at each step. They will differ slightly from the paper. **If your final cohort is far below ~150 patients, investigate before training** (the earlier 48/6/12 split was too small).

### 3.3 Splits
- **Design A (paper-style):** patient-level stratified split targeting 111/15/30. Repeat with at least 3 seeds. Version `splits.json` + seeds.
- **Design B (CV):** 5-fold patient-level stratified CV on all eligible patients, used for the radiomics-only model and vViT. Validation for early stopping comes from a patient-level slice of the training folds.
- **Always split by patient first.** Never split slices randomly.
- Balance classes by subsampling images in **train only**. Do not subsample test or validation folds; report on the true patient prevalence (the paper's image-balanced test set is a deviation we avoid).

### 3.4 Leakage rules (non-negotiable)
- Radiomic top-16 selection (ANOVA F-value) and z-score statistics are fit on **training data only**, **inside each fold or seed**.
- Freeze the feature list before touching validation or test data.
- No hyperparameter tuning on test data.

### 3.5 Decision to make and document
The paper does not say whether radiomics are computed **per slice** (2D crop) or **per whole tumor** (3D). Per-tumor features give every slice of a patient the same vector, which makes patient memorization easier. Pick one, state it in the methods, and keep it fixed.

---

## 4. Models

### M0: Radiomics-only sanity gate (do this first)
- Logistic regression and random forest on the 16 radiomic features, patient-level 5-fold CV, top-16 selection inside each fold.
- **Gate (heuristic):** if patient-level AUROC is near chance (~0.5–0.6), audit labels, extraction and leakage before training any neural model. The paper reports radiomics as the dominant signal, so some signal should appear.

### M1: vViT (core)
| Setting | Value |
|---|---|
| Embedding dim | 128 |
| Heads / head dim | 8 / 64 (inner attention dim 512, projected back to 128; requires a custom attention block, not `nn.TransformerEncoder`) |
| MLP dim / depth | 32 / 8 |
| Tokens | class, demographic, comorbidity, habit, radiomic, image (per-sector linear tokenizers) |
| Heads | One linear classifier per output token (6 total) |
| Fusion | Majority vote over 6 sector predictions (3–3 ties broken by class token; our choice) |
| Loss | BCE per head, mean over batch × heads (our aggregation choice) |
| Optimizer | Adam, β=(0.9, 0.999), ε=1e-8, weight decay 0, AMSGrad off; lr 1e-3 (paper does not state; our choice) |
| Augmentation | Image sector only: h-flip, v-flip, perspective, invert, posterize, solarize, equalize |
| Training | Up to 200 epochs, early stopping (patience 15 on val accuracy), AMP, save best-val checkpoint |

Note: because every output token attends to all inputs, per-sector heads are **not independent per-sector models**. Interpret sector importance via permutation importance, not per-sector accuracy.

### M2 (optional): one image-only baseline
- One pretrained timm model (ResNet18 or ConvNeXt-tiny), lr 1e-4, same augmentation, paper-style split only (not CV).
- Add DeLong/McNemar comparison only if M2 is trained.
- Do not run the full ViT / ConvNeXt / ResNeXt trio unless time allows.

---

## 5. Evaluation

### 5.1 Metrics
Accuracy, sensitivity, specificity, PPV, NPV, F-score, log loss, Cohen's κ, AUROC.

### 5.2 Patient-level aggregation (primary)
Mean slice probability per patient (for AUROC) and majority vote (for class labels).

### 5.3 Confidence intervals
- **Resample patients**, not slices, for bootstrap (~1000 resamples).
- For CV: report mean ± SD across folds and pooled out-of-fold AUROC.
- Report image-level metrics only as secondary, labelled "inflated by within-patient correlation".

### 5.4 Permutation importance
1. Baseline accuracy of the trained model on the test set.
2. Per sector: assign each patient's sector values to another patient (shuffle across patients, other sectors intact) and recompute accuracy.
3. Repeat 100× per sector. Importance = drop in accuracy.
4. Pairwise Mann–Whitney U between sectors; boxplots.

Paper reference: radiomic sector dominant (about 0.30 image-level and 0.154 patient-level); image sector near zero or negative.

### 5.4 Comparison context
Paper patient-level test results for reference: accuracy 0.733, AUROC 0.795 (95% CI 0.594–0.962). Treat as context, not a pass/fail threshold.

---

## 6. Speed and training-time checklist

An estimated 37 h run indicates a bug or misconfiguration. Expected: vViT ~15 min total; a pretrained CNN baseline ~1 h. Check in order:

1. `torch.cuda.is_available()` is `True` (the Python 3.14 / torch 2.14 stack may be CPU-only).
2. Time one pass through the DataLoader with no model, then one forward/backward pass on a single batch. If data loading dominates, tune `num_workers` (Windows: 2–4, `persistent_workers=True`) and cache preprocessed crops as tensors/`.npy` instead of reading and augmenting from disk.
3. Confirm the epoch uses the balanced ~1.4k training crops, not every tumor slice.
4. Confirm AMP is actually enabled and tensors are on GPU.
5. Use small baselines (ResNet18) with early stopping; do not run CV on CNN baselines.

---

## 7. Demo (after training)

A small `demo.py` for a **test-split** case:
1. Load CT + tumor mask + `kits.json`.
2. Crop slices, compute radiomics (using the frozen top-16 list and train z-score stats), build clinical vector.
3. Run vViT; print per-sector votes, final vote and P(high grade) per slice and per patient.

**Input requirement:** the model needs the image crop, the tumor mask (for radiomics) and the clinical record. A raw CT alone is not enough; segmenting a new CT would require a separate model and is out of scope. Present the demo as a research illustration, not a diagnostic tool.

---

## 8. Task checklist and milestones

| # | Milestone | Status |
|---|---|---|
| M0 | Data downloaded, explored, literature reviewed | ✅ done |
| M1 | Fix cohort: real clinical fields, ~157–172 patients, new `splits.json` | ☐ |
| M2 | Radiomics-only CV sanity gate (§4 M0) | ☐ |
| M3 | Speed fix (§6): full vViT run in ≲1 h | ☐ |
| M4 | vViT on paper-style split (≥3 seeds) + 5-fold CV, patient-level metrics + bootstrap CIs | ☐ |
| M5 | Permutation importance + figures | ☐ |
| M6 | (Optional) one image baseline + DeLong/McNemar | ☐ |
| M7 | Demo script | ☐ |
| M8 | Report | ☐ |

---

## 9. Deviations to declare in the report
- Python / PyTorch version differs from the paper (3.8 / 1.7.1).
- Cohort counts differ from 157 → 111/15/30 (source and seed differences).
- DM definition and alcohol-level mapping (§3.1).
- Radiomics per slice vs per tumor (§3.5).
- Loss aggregation across heads, learning rate, and 3–3 vote tie-break are under-specified in the paper; ours are documented.
- Early stopping + AMP added for compute budget (paper trains a fixed 200 epochs in fp32).
- Class balancing applied to train only; test sets keep true patient prevalence.
- Patient-level CV in addition to the paper's single split.
- Baseline pretraining status and lr (1e-4; lr 1e-3 collapsed pretrained backbones).
- Data source: GitHub KiTS19 mirror vs TCIA C4KC-KiTS.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Tiny test set → huge variance | Patient-level CV, multiple seeds, patient-level bootstrap CIs |
| Weak signal in the data | M0 gate; report honestly, as the paper's own CIs are wide |
| Leakage via feature selection or slices | §3.4 rules; split by patient first |
| Slow training | §6 checklist |
| Overfitting (train loss ≈ 0 early) | Early stopping, augmentation, smaller epoch cap |
| Paper details unspecified | Declare choices in §9 |

---

## 11. Repo housekeeping

**Keep:** data download and exploration, SimpleITK I/O, cropping, PyRadiomics extraction, dataset class, vViT / sectors / voting code, metrics and bootstrap, DeLong and McNemar functions (only if M6).

**Rewrite or update:** `clinical.py` (real kits.json fields; remove synthetic columns), `dataset_split.py` (patient-level, train-only balancing, CV folds), evaluation (patient-level bootstrap), vViT attention (head dim 64).

**Retire or move to a branch:** `synthetic_clinical.py`, cross-attention fusion improvement, full three-baseline suite, hyperparameter search, 3D CNN encoder.

**Docs:** one-page README with the pipeline diagram in §2; `plan.md` (this file); `LOG.md` for dated results and decisions.

---

## 12. Notes on the paper (for the report)
- The paper's Table 1 appears to have the High/Low grade rows swapped relative to its text. Text: training 75 low / 36 high. Use the text.
- Per-sector accuracies are all ~0.8 (even demographics-only: 0.838), because output tokens attend to all inputs. Do not read them as independent sector performance.
- The paper concludes raw CT images added little (negative permutation importance); radiomics carried the signal.
- Image-level test set was balanced by subsampling images, not patients (23 low / 7 high patients), so image-level metrics should not be compared with ours directly.
