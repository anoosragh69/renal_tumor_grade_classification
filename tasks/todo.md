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

> ✅ **Real data re-run 2026-09-23:** Steps 2–5 on `kits19/data/` (210 with segs) → split **train 48 / val 6 / test 12** patients (9467/1178/2230 slices). Radiomics still on mock features — **pyradiomics 3.1.0 now installed** in Miniconda env `radiomics` (Python 3.9; no cp314 wheels); full extraction script ready (`--workers`, resume checkpoints) but **not yet executed** (takes ~50 min on 7 workers). Earlier 40-case mock pilot (8/2/3) superseded for real-data work.

- [x] Step 2: Segmentation-guided 2D cropping (Fig. 1, 128×128 LANCZOS)
- [x] Step 3: PyRadiomics extraction + top-16 F-value selection (train-only) — script ready for real run (conda env + parallel/resume); current CSVs still mock until full extraction executes
- [x] Step 4: Synthetic clinical fields (Table 1 prevalences, label-independent)
- [x] Step 5: Exclusion criteria + stratified split → `splits.json` + seed
- [x] Step 6: Sector-dict Dataset/DataLoader + paper augmentations (commit `190b20f`)
- [x] **Checkpoint:** pipeline runs end-to-end on pilot subset ✅ (2026-09-22) — `splits.json` + pilot sector-dict DataLoader ready for handoff to B

## M2: vViT Sanity Check (Step 7)

- [x] Step 7a: Sector tokenizers + class token + transformer encoder (+ per-sector heads) — `src/models/vvit.py` skeleton, dummy self-test passed
- [x] Step 7b: Majority-voting fusion — `src/models/fusion_baseline_vote.py` (≥4/6 majority; class-token tie-break)
- [x] Step 7c: BCE multi-sector loss + Adam (paper hyperparams) — `src/training/train.py`
- [x] **Checkpoint:** model overfits a tiny subset ✅ (2026-09-23) — 16-slice subset, loss 0.52→0.0014 over 60 epochs; also verified on real kits19 batch (train 48 / val 6 / test 12; radiomics mock-mode)

## M3: Full Training (Steps 7–9)

- [ ] Step 7d: Full 200-epoch training run, best-val checkpoint
- [x] Step 9a scaffolding: metrics + bootstrap CI + patient-level aggregation — `src/training/evaluation.py` (full run pending trained model)
- [ ] **Checkpoint:** paper-style metrics table reproduced

## M4: Baselines & Statistical Tests (Steps 8–9)

- [ ] Step 8: timm ViT / ConvNeXt / ResNeXt (2D image-only) trained
- [x] Step 9b: DeLong test (custom implementation) — `src/stats_tests.py`, self-test passed
- [x] Step 9c: McNemar test — `src/training/evaluation.py::mcnemar_test` (statsmodels + fallback)
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
- **Current status:** M0 + M1 complete (2026-09-22); M2 complete (2026-09-23) — vViT skeleton + voting + BCE/Adam + overfit sanity on real kits19 split (48/6/12 patients; 9467/1178/2230 slices).
- **Dataset:** Real `kits19/data/` present (210 cases with segmentation + kits.json; cases 00210–00299 imaging-only). Steps 2–5 re-run 2026-09-23 → `splits.json` + `top16_features.json` updated. **Radiomics CSVs still mock** — real extraction ready but pending run.
- **Radiomics env:** pyradiomics 3.1.0 in Miniconda env `radiomics` (`%USERPROFILE%\miniconda3\envs\radiomics\python.exe`; Python 3.9, numpy 1.26.4 — no cp314 wheels for pyradiomics). Full run: that python + `src/data/radiomics_extraction.py --splits data/processed/splits.json --workers 7` (~50 min; per-patient resume checkpoints in `data/processed/radiomics_parts/`). `normalize=True` + binWidth=25 gives ~0.4 s/slice (~20× vs raw).
- **Environment:** torch 2.14.0+**cu126** + torchvision 0.29.0+cu126 (manual wheels from cu126 index; cu128 lacks 2.14.0 for py3.14). `torch.cuda.is_available()=True` on RTX 3050 Laptop 6GB.
