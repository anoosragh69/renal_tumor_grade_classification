# Task Checklist: vViT Renal Tumor Grade Classification (Integrated Plan)

Authoritative plan: `tasks/plan.md` (integrates `references/plan.md`)

## M0: Data, Exploration & Literature ✅ COMPLETE

- [x] Task 1: Literature Review - Paper Collection
- [x] Task 2: Literature Summary Table
- [x] Task 3: 1-Page Paper Summary
- [x] Task 4: Complete Dataset Download (300 cases + segmentations)
- [x] Task 5: Data Exploration & Analysis
- [x] Step 1: Pin `requirements.txt` (commit `76dfd30`)
- [x] Step 1: Audit WHO/ISUP grade availability (commit `833b0d4`) — 172/210 labeled (low 119 / high 53), grades 1–4 only, all labeled cases have masks; GitHub mirror vs TCIA deviation documented in `docs/isup_grade_audit.md`

## M1: Pilot Data Pipeline (Steps 2–6, ~20-patient subset)

> ⚠️ **Dataset pending:** Real KiTS19 path not yet provided. Steps 2–6 validated end-to-end on a **40-case seeded mock pilot** (`scratch/generate_pilot_data.py`, seed 42): exclusions exercised (4 null-grade, 22 slice-count outliers), equalise + stratified split → train 8 / val 2 / test 3; Step 6 self-test + handoff check passed. Re-run each script against real `kits19/data/` once path is confirmed.

- [x] Step 2: Segmentation-guided 2D cropping (Fig. 1, 128×128 LANCZOS)
- [x] Step 3: PyRadiomics extraction + top-16 F-value selection (train-only)
- [x] Step 4: Synthetic clinical fields (Table 1 prevalences, label-independent)
- [x] Step 5: Exclusion criteria + stratified split → `splits.json` + seed
- [x] Step 6: Sector-dict Dataset/DataLoader + paper augmentations (commit `190b20f`)
- [x] **Checkpoint:** pipeline runs end-to-end on pilot subset ✅ (2026-09-22) — `splits.json` + pilot sector-dict DataLoader ready for handoff to B

## M2: vViT Sanity Check (Step 7)

- [ ] Step 7a: Sector tokenizers + class token + transformer encoder
- [ ] Step 7b: Per-sector heads + majority-voting fusion
- [ ] Step 7c: BCE multi-sector loss + Adam (paper hyperparams)
- [ ] **Checkpoint:** model overfits a tiny subset

## M3: Full Training (Steps 7–9)

- [ ] Step 7d: Full 200-epoch training run, best-val checkpoint
- [ ] Step 9a: Full metrics + bootstrap 95% CI + patient-level aggregation
- [ ] **Checkpoint:** paper-style metrics table reproduced

## M4: Baselines & Statistical Tests (Steps 8–9)

- [ ] Step 8: timm ViT / ConvNeXt / ResNeXt (2D image-only) trained
- [ ] Step 9b: DeLong test (custom implementation)
- [ ] Step 9c: McNemar test
- [ ] **Checkpoint:** DeLong/McNemar comparison table produced

## M5: Explainability (Step 10)

- [ ] Step 10: Permutation feature importance (100× per sector)
- [ ] Step 10b: Mann–Whitney U pairwise + Fig. 5-style boxplots
- [ ] Step 9d: Fig. 5-style ROC curves
- [ ] **Checkpoint:** Fig. 4/5 equivalents reproduced

## M6: Improvement — Attention Fusion (Step 11)

- [ ] Step 11a: `fusion_improved.py` — class-token head
- [ ] Step 11b: Attention-weighted sector fusion
- [ ] Step 11c: Attention vs permutation-importance ranking comparison
- [ ] **Checkpoint:** voting vs class-token vs attention fusion results

## M7: Report (Step 12)

- [ ] Step 12: Report draft (Intro → Methods → Results → Discussion → Improvement)
- [ ] Declare all deviations (Section 6 of plan)
- [ ] **Checkpoint:** final writeup complete

---

## Notes
- **Superseded work:** old Tasks 6–15 (3D CNN / early-fusion pipeline) are replaced by the integrated plan — see mapping table in `tasks/plan.md` §8.
- **Reusable code:** SimpleITK I/O, kits.json loading, metrics, trainer/baseline skeletons — disposition table in `tasks/plan.md` §1.
- **Current status:** M0 + M1 complete (2026-09-22) — Steps 2–6 run end-to-end on a 40-case seeded mock pilot (8/2/3 split); handoff to B ready (`splits.json`, `get_dataloaders()`).
- **Dataset pending:** User will provide real `kits19/data/` path — re-run Steps 2–6 for full validation (pilot artifacts in `kits19/`, `data/interim/`, `data/processed/` are gitignored except `splits.json` + `top16_features.json`).
- **Environment:** torch 2.14.0+**cpu** installed (PyPI default wheel; `torch.cuda.is_available()=False` despite RTX 2050). For B's 200-epoch runs, reinstall CUDA build: `pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu128` (verify cuXXX matches driver).
