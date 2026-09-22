# Teammate Split: vViT Renal Tumor Grade Classification

Divides `todo.md` / `tasks/plan.md` between 2 teammates. Shared milestones stay coupled: **M1 blocks M2**, so parallelize the pilot phase, then converge.

---

## Teammate A — Data & Explainability Pipeline

**Owns:** data acquisition → splits (Steps 1–6), explainability (Step 10), report data/results sections

### M0 leftovers
- [x] Step 1: Pin `requirements.txt` (commit `76dfd30`)
- [x] Step 1: Audit WHO/ISUP grade availability (commit `833b0d4`) — 172 usable labels, see `docs/isup_grade_audit.md`

### M1: Pilot Data Pipeline (~20-patient subset)
> ⚠️ **Dataset pending:** Real KiTS19 path not yet provided. Steps 2–6 implemented and tested on mock data. Re-run once path is confirmed.
- [x] Step 2: Segmentation-guided 2D cropping (Fig. 1, 128×128 LANCZOS)
- [x] Step 3: PyRadiomics extraction + train-only top-16 F-value selection
- [x] Step 4: Synthetic clinical fields (Table 1 prevalences, label-independent, `synthetic=True` flag)
- [ ] Step 5: Exclusion criteria + stratified split → `splits.json` + seed
- [ ] Step 6: Sector-dict Dataset/DataLoader + paper augmentations
- [ ] **M1 checkpoint:** pipeline runs end-to-end on pilot subset → hand off to B

### M5: Explainability (after M3 model exists)
- [ ] Step 10a: Permutation feature importance (100× per sector)
- [ ] Step 10b: Mann–Whitney U pairwise + Fig. 5-style boxplots
- [ ] Step 9d: Fig. 5-style ROC curve plots *(data/plotting side)*

### M7: Report
- [ ] Methods: data, radiomics, synthetic fields, splits
- [ ] Results: Tables 1–2 equivalents, Fig. 4–5 equivalents
- [ ] Limitations: synthetic clinical data, source/seed count differences

---

## Teammate B — Model, Training & Statistics

**Owns:** vViT + baselines (Steps 7–8), training/eval/stats (Step 9), improvement (Step 11), report model/results sections

### Parallel prep (can start before M1 finishes)
- [ ] Step 7 skeleton: sector tokenizers, class token, transformer encoder, per-sector heads (test with dummy tensors)
- [ ] Step 9 scaffolding: metrics + bootstrap CI, McNemar, Mann–Whitney wrappers
- [ ] DeLong test implementation *(independent of data — develop on synthetic predictions)*

### M2: vViT Sanity Check (needs A's M1 pilot data)
- [ ] Step 7b: Majority-voting fusion head
- [ ] Step 7c: BCE multi-sector loss + Adam (paper hyperparams: β1=0.9, β2=0.999, ε=1e-8, wd=0)
- [ ] **M2 checkpoint:** model overfits a tiny subset

### M3: Full Training
- [ ] Step 7d: Full 200-epoch run, best-val checkpoint, TensorBoard logging
- [ ] Step 9a: Full metrics + patient-level aggregation
- [ ] **M3 checkpoint:** paper-style metrics table

### M4: Baselines & Statistical Tests
- [ ] Step 8: timm ViT / ConvNeXt / ResNeXt (2D image-only), same protocol
- [ ] Step 9b–c: DeLong + McNemar comparison table
- [ ] **M4 checkpoint:** baseline comparison table

### M6: Improvement — Attention Fusion
- [ ] Step 11a: `fusion_improved.py` — class-token head
- [ ] Step 11b: Attention-weighted sector fusion
- [ ] Step 11c: Attention vs permutation-importance ranking comparison *(feeds A's M5 output)*

### M7: Report
- [ ] Methods: model architecture, training, stats
- [ ] Results: Table 3 equivalents, DeLong/McNemar
- [ ] Improvement section + deviations (PyTorch version, loss aggregation, baseline pretraining)

---

## Handoff / Sync Points

| When | From → To | Artifact |
|---|---|---|
| M0 | A → B | Pinned `requirements.txt`, grade-availability audit |
| M1 done | A → B | `splits.json`, pilot sector-dict DataLoader |
| M2–M4 | B → A | Trained checkpoint(s) + test-set predictions/probabilities |
| M5 + M6 | A ↔ B | Permutation-importance rankings vs attention rankings |
| M7 | Both | Assemble `report/` — A: data/results, B: model/improvement |

## Shared responsibility
- [ ] Git hygiene: small focused commits, no data in repo
- [ ] Weekly milestone sync against `tasks/todo.md`
