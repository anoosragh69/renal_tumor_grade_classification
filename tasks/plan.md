# Implementation Plan: vViT Renal Tumor Grade Classification (Integrated)

**Source paper:** Usuzaki et al., *Multimodal Deep Learning for Predicting WHO/ISUP Grading in Renal Tumors on CT*, European Journal of Radiology Open 17 (2026) 100796
**Authoritative reference:** `references/plan.md` — this document integrates it with current project state and completed work.

---

## 0. Scope Decisions (locked)

| Decision | Choice |
|---|---|
| Compute | Local GPU workstation |
| Data | Full replication — real KiTS19 CT + PyRadiomics-derived radiomics; clinical comorbidity/habit fields **synthetic** (option a in reference plan), clearly labeled as such |
| Task | Binary low/high WHO-ISUP grade (replaces earlier 4/5-class formulation) |
| Improvement track | Cross-sector attention fusion + attention-based explainability (replaces the paper's naive majority-voting head) |

**Replication caveat (to document in report):** KiTS19 does not expose comorbidity/smoking/alcohol data. These fields are synthesized from population-level prevalence rates (paper Table 1), sampled **independent of the label** so they act as noise sectors — which also lets us validate that permutation importance correctly assigns them low importance.

---

## 1. Current Project State

### Complete (carried over)
- All 300 KiTS19 cases downloaded (`kits19/data/`, case_00000–00299) with segmentations
- Data exploration: ISUP distribution, missingness, image-dimension statistics
- Literature review: 10 papers + summary table + 1-page paper summary (`docs/`)
- Repo scaffolding, git history, README

### Code status — adapt, don't discard

| Existing file | Disposition |
|---|---|
| `src/data/preprocess.py` | **Partially keep** — SimpleITK I/O (`load_nifti`, resampling); rewrite cropping to per-slice min-rectangle 2D (Fig. 1) |
| `src/data/clinical.py` | **Rewrite** — keep kits.json loading; replace feature schema with 3-sector (demographic/comorbidity/habit) + synthetic fields + binary label |
| `src/data/dataset.py` | **Rewrite** — sector-dict samples instead of `(volume, clinical, label)` tuples; 2D image tensor `(1,128,128)`; paper's augmentations |
| `src/models/image_encoder.py` | **Retire** (3D CNN) — replaced by linear image-sector projection; keep only as optional ablation if time permits |
| `src/models/clinical_encoder.py` | **Retire** — replaced by per-sector linear tokenizers |
| `src/models/multimodal.py` | **Replace** with `vvit.py` + sector tokenizers + voting head |
| `src/training/trainer.py` | **Rewrite** — BCE multi-sector loss, Adam (paper hyperparams), 200 epochs, sector metrics |
| `src/training/metrics.py` | **Extend** — bootstrap CI, patient-level aggregation, log loss, Cohen's κ |
| `src/training/baselines.py` | **Rewrite** — image-only ViT/ConvNeXt/ResNeXt (timm), 2D input |
| `src/training/hparam_search.py` | **Drop** — paper specifies hyperparameters; not needed for replication |

**Missing entirely (new work):** radiomics pipeline, synthetic clinical fields, split/exclusion pipeline, vViT model, voting head, permutation importance, DeLong/McNemar/Mann-Whitney statistics, cross-attention fusion improvement.

---

## 2. Stack

```
torch, torchvision, pyradiomics, SimpleITK, scikit-learn,
statsmodels, scipy, pandas, numpy, matplotlib, seaborn,
timm, tqdm, pyyaml, tensorboard (or wandb)
```

- Python 3.10+, PyTorch ≥2.1 (note deviation from paper's 1.7.1)
- Data stays out of git; `splits.json` + seeds are versioned

---

## 3. Target Repository Structure

```
├── configs/                    # data.yaml, model.yaml, train.yaml  [NEW]
├── data/
│   ├── raw/                    # → symlink/use existing kits19/data/
│   ├── interim/                # cropped 2D tumor patches
│   └── processed/              # tensors, radiomics features, splits.json
├── src/
│   ├── data/
│   │   ├── segment_and_crop.py       # [NEW] Fig.1 min-rect crop, 128×128 LANCZOS
│   │   ├── radiomics_extraction.py   # [NEW] PyRadiomics, top-16 F-value selection
│   │   ├── synthetic_clinical.py     # [NEW] comorbidity/habit/BMI synthesis
│   │   ├── dataset_split.py          # [NEW] exclusions + stratified 111/15/30 split
│   │   ├── clinical.py               # [REWRITE] kits.json → demographics/label
│   │   └── dataset.py                # [REWRITE] sector-dict Dataset
│   ├── models/
│   │   ├── vvit.py                   # [NEW] core multi-sector transformer (Fig.3)
│   │   ├── sectors.py                # [NEW] per-sector embeddings + linear heads
│   │   ├── fusion_baseline_vote.py   # [NEW] paper's majority-voting head
│   │   ├── fusion_improved.py        # [NEW] cross-attention fusion (improvement)
│   │   └── baselines.py              # [REWRITE] 2D ViT/ConvNeXt/ResNeXt
│   ├── train.py                      # [REWRITE] from trainer.py
│   ├── evaluate.py                   # [EXTEND] from metrics.py
│   ├── permutation_importance.py     # [NEW]
│   ├── stats_tests.py                # [NEW] DeLong, McNemar, Mann-Whitney, bootstrap
│   └── utils.py
├── notebooks/                # exploration only (existing notebooks stay)
├── results/                   # metrics/, figures/, checkpoints/
├── report/
├── tasks/plan.md             # this file
├── references/plan.md        # authoritative source plan
├── requirements.txt          # [NEW] pin versions
└── README.md
```

---

## 4. Step-by-Step Implementation

### Step 1 — Environment & data acquisition ✅ MOSTLY COMPLETE
- [x] 300 cases downloaded with segmentations (210 with masks + kits.json metadata)
- [x] kits.json clinical metadata present
- [x] Pin `requirements.txt` — Python 3.14 stack, torch 2.14.0 (commit `76dfd30`)
- [x] Audit WHO/ISUP grade availability — **172/210 labeled** (low 119 / high 53; grades 1–4, no 5); all labeled cases have segmentations; source = GitHub kits19 mirror (deviation vs paper's TCIA C4KC-KiTS, declare in §6) — see `docs/isup_grade_audit.md` (commit `833b0d4`)

**Checkpoint:** load a case's CT + mask and view a middle slice ✅
**Step 1 status: ✅ COMPLETE (2026-09-22)**

### Step 2 — Segmentation-guided cropping (Fig. 1) [NEW]
1. For every axial slice containing tumor voxels: compute min bounding rectangle around the tumor
2. Crop to that rectangle, resize to **128×128** with LANCZOS (`PIL.Image.resize(..., Image.LANCZOS)`)
3. Save cropped tumor-only 2D patches per slice, indexed by `(patient_id, slice_id)`, to `data/interim/`

### Step 3 — Radiomics extraction [NEW]
1. Run PyRadiomics on each 2D crop + mask: enable all feature classes (~105 features — first-order, shape, GLCM, GLRLM, GLSZM, GLDM, NGTDM)
2. **Train split only:** rank features by ANOVA F-value (`sklearn.feature_selection.f_classif`) against the binary label
3. Select the **top 16** features; freeze that list for val/test (never re-select on val/test — leakage guard)
4. Z-score normalize using train-set mean/std

### Step 4 — Synthetic clinical fields [NEW]
1. Real: age, sex from kits.json; BMI real if present, else sample ~N(30.7, appropriate σ)
2. Synthetic: PVD, DM, CKD, smoking, alcohol — Bernoulli/categorical draws at paper Table 1 prevalences, **independent of label**
3. Flag columns with `synthetic=True` in the dataframe; state in report limitations

### Step 5 — Exclusions & dataset split [NEW]
Reproduce the paper's Fig. 2 pipeline programmatically:
1. Exclude patients missing any required clinical field (≈0 after synthesis)
2. Exclude patients missing WHO/ISUP grade
3. Exclude patients whose image count falls outside mean ± 1 SD
4. Exclude radiomic extraction failures or tumor masks < 256 pixels
5. Random-select to equalize image counts per class within each split
6. Stratified random split targeting ~111/15/30 patients (document actual counts — they will differ)

**Deliverable:** `data/processed/splits.json` with patient IDs per split + logged seed

### Step 6 — Dataset / DataLoader [REWRITE] ✅
Sector-dict sample:
```python
{
  "demographic": [age, sex, bmi],        # len 3
  "comorbidity": [pvd, dm, ckd],         # len 3
  "habit": [smoking, alcohol],           # len 2
  "radiomic": [16 features],             # len 16
  "image": tensor(1, 128, 128),          # flattened to 16384 in model
  "label": 0/1
}
```
Augmentations (train split, **image sector only**): horizontal flip, vertical flip, perspective, invert, posterize, solarize, equalize (via `torchvision.transforms`)

**Status:** ✅ complete (commit `190b20f`) — `src/data/dataset.py` rewritten as `RenalTumorSectorDataset` + `get_dataloaders()`; self-test verifies schema, train-only augmentation, val/test determinism, label consistency, split disjointness. Sample also carries `patient_id`/`slice_idx` metadata for patient-level aggregation (Step 9) and permutation importance (Step 10).

### Step 7 — vViT model (Fig. 3) [NEW — core of project]
1. **Sector tokenizers:** linear projection per sector into shared embedding dim
   - embedding dim: **128** · heads: **8** (head dim 64) · MLP dim: **32** · depth: **8**
2. **Class token:** learnable embedding prepended to the sector sequence
3. **Sequence:** `[class, demographic, comorbidity, habit, radiomic, image]` → `(batch, 6, 128)`
4. **Transformer encoder:** pre-norm MHA + MLP blocks × 8 (manual implementation or customized `nn.TransformerEncoder`)
5. **Per-sector heads:** each of the 6 output tokens → own linear classifier → 6 sector logits (enables Table 2 per-sector metrics)
6. **Baseline fusion:** majority voting across the 6 sector predictions (implement first to reproduce paper numbers)
7. **Loss:** BCE per sector head, joint backprop through shared encoder (paper under-specifies aggregation — document our choice)
8. **Optimizer:** Adam, β1=0.9, β2=0.999, ε=1e-8, weight_decay=0, AMSGrad=False (exact paper match)
9. **Training:** 200 epochs, save best-validation-accuracy checkpoint

### Step 8 — Baseline comparison models [REWRITE]
timm ViT / ConvNeXt / ResNeXt, image-only, 2D 128×128 input, same 200 epochs / same augmentation / same train-val split → Table 3 comparison. Note pretrained-vs-scratch choice as a deviation (paper doesn't specify).

### Step 9 — Evaluation [EXTEND]
1. Image-based metrics: accuracy, sensitivity, specificity, PPV, NPV, F-score, log loss, Cohen's κ, AUC-ROC — each with bootstrap 95% CI (~1000× test-set resamples)
2. Patient-level aggregation: majority vote for binary label; mean probability for AUC
3. **DeLong test** for AUC comparison vs baselines (custom U-statistic implementation — budget real time; no standard library ships this)
4. McNemar (`statsmodels.stats.contingency_tables.mcnemar`)
5. Reproduce Fig. 5 ROC curves + permutation-importance boxplots

### Step 10 — Permutation feature importance (Fig. 4) [NEW]
1. Baseline test accuracy with trained model
2. Per sector: shuffle values across patients (others intact), recompute accuracy, record drop
3. Repeat 100× per sector with different shuffles
4. Pairwise Mann–Whitney U across sectors → boxplots with significance brackets (Fig. 5b/d style)

### Step 11 — Improvement: cross-sector attention fusion [NEW]
1. `fusion_improved.py`: class-token final-layer output → single classification head (standard ViT/BERT-style pooling vs voting)
2. Extract last-layer class-token attention over the 5 non-class sectors, average across heads/samples → learned sector-importance ranking
3. Compare attention ranking vs permutation-importance ranking (novel analysis: do the two importance measures agree?)
4. Report: voting vs class-token vs attention-weighted fusion performance

### Step 12 — Report [NEW]
Structure mirroring the paper: Intro → Methods (data, model, stats) → Results (Tables 1–3 equivalents, Figs 4–5 equivalents) → Discussion (deviations below) → Improvement section.

---

## 5. Milestones

| # | Checkpoint | Status |
|---|---|---|
| M0 | Data downloaded + exploration + literature review | ✅ done |
| M1 | Pipeline end-to-end on ~20-patient pilot (crop → radiomics → synthetic → split) | pending |
| M2 | vViT overfits a tiny subset (architecture + loss sanity check) | pending |
| M3 | Full training run; paper-style metrics table reproduced | pending |
| M4 | Baselines (ViT/ConvNeXt/ResNeXt) trained; DeLong/McNemar table | pending |
| M5 | Permutation importance + Fig. 5-style plots | pending |
| M6 | Attention fusion improvement implemented, compared vs voting | pending |
| M7 | Report finalized | pending |

---

## 6. Deviations to Declare in Report
- Synthetic comorbidity/habit variables (sampled independent of label — not real patient data)
- PyTorch version ≠ paper's 1.7.1
- Patient counts after exclusions ≠ paper's 157→111/15/30 (source/seed differences)
- Loss aggregation across sector heads under-specified in paper — document our choice
- Baseline pretraining status (timm pretrained vs random init) — state explicitly
- Data source: kits19 GitHub mirror vs C4KC-KiTS TCIA (affects grade availability)

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| WHO/ISUP grades missing/incomplete in kits.json | High | Audit early in Step 1; fall back to TCIA pull if needed |
| PyRadiomics install issues (Windows) | High | Test on 1 case first; use provided wheels/conda |
| DeLong implementation time | Medium | Base on known open-source U-statistic snippet |
| 200-epoch runs long | Medium | TensorBoard logging; small-epoch pilot runs first |
| Synthetic sectors add label-independent noise | Low (by design) | Validates permutation-importance low-importance finding |
| Small dataset → overfitting | Medium | Paper's augmentations, best-val checkpointing |

---

## 8. Old → New Task Mapping

| Old tasks (`tasks/todo.md`) | New plan step | Status |
|---|---|---|
| Tasks 1–3 (literature review) | Report prep | ✅ complete — carried over |
| Task 4 (dataset download) | Step 1 | ✅ complete — carried over |
| Task 5 (data exploration) | Step 1 audit | ✅ complete — carried over |
| Task 6 (3D CT preprocessing) | Step 2 (2D crop — rewrite) | Superseded |
| Task 7 (clinical preprocessing) | Steps 4–5 (schema rewrite) | Superseded |
| Tasks 8–10 (3D CNN / MLP / early fusion) | Steps 7–8 (vViT + baselines) | Superseded |
| Task 11 (Dataset/DataLoader) | Step 6 (sector-dict rewrite) | Superseded |
| Task 12 (training pipeline) | Step 7 (BCE multi-sector rewrite) | Superseded |
| Task 13 (evaluation metrics) | Step 9 (extend) | Partially reused |
| Task 14 (hparam search) | — | Dropped (paper fixes hyperparameters) |
| Task 15 (baselines) | Step 8 (rewrite as 2D image-only) | Superseded |
| — | Steps 3, 10, 11 (radiomics, permutation importance, attention fusion) | New work |
