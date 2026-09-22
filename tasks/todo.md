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

- [x] Step 2: Segmentation-guided 2D cropping (Fig. 1, 128×128 LANCZOS)
- [ ] Step 3: PyRadiomics extraction + top-16 F-value selection (train-only)
- [ ] Step 4: Synthetic clinical fields (Table 1 prevalences, label-independent)
- [ ] Step 5: Exclusion criteria + stratified split → `splits.json` + seed
- [ ] Step 6: Sector-dict Dataset/DataLoader + paper augmentations
- [ ] **Checkpoint:** pipeline runs end-to-end on pilot subset

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
- **Current status:** M0 complete; starting M1.
