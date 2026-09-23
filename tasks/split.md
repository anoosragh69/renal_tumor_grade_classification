# Teammate Split: vViT Renal Tumor Grade Classification

Divides `todo.md` / `tasks/plan.md` between 2 teammates. Shared milestones stay coupled: **M1 blocks M2**, so parallelize the pilot phase, then converge.

---

## Teammate A — Data & Explainability Pipeline

**Owns:** data acquisition → splits (Steps 1–6), explainability (Step 10), report data/results sections

### M0 leftovers
- [x] Step 1: Pin `requirements.txt` (commit `76dfd30`)
- [x] Step 1: Audit WHO/ISUP grade availability (commit `833b0d4`) — 172 usable labels, see `docs/isup_grade_audit.md`

### M1: Pilot Data Pipeline (~20-patient subset)
> ✅ **Real data re-run 2026-09-23 (B):** Steps 2–5 executed against `kits19/data/` — 210 cases with segmentation cropped; clinical 172 labeled; split **train 48 / val 6 / test 12** patients (9467/1178/2230 slices); radiomics **now real PyRadiomics** — full extraction executed 2026-09-23 in conda env `radiomics` (all 210 cases; top-16 re-selected train-only). Earlier mock pilot (40 cases → 8/2/3) superseded for real-data work.
- [x] Step 2: Segmentation-guided 2D cropping (Fig. 1, 128×128 LANCZOS)
- [x] Step 3: PyRadiomics extraction + train-only top-16 F-value selection
- [x] Step 4: Synthetic clinical fields (Table 1 prevalences, label-independent, `synthetic=True` flag)
- [x] Step 5: Exclusion criteria + stratified split → `splits.json` + seed
- [x] Step 6: Sector-dict Dataset/DataLoader + paper augmentations (commit `190b20f`)
- [x] **M1 checkpoint:** pipeline runs end-to-end on pilot subset → hand off to B ✅ (2026-09-22) — artifact: `splits.json` (train 8 / val 2 / test 3) + `get_dataloaders()` in `src/data/dataset.py`; handoff-check passed (all 3 loaders batch, `SECTOR_DIMS` exported)
- [x] **Real-data handoff:** Steps 2–5 re-run on kits19 by B (2026-09-23) → `splits.json` 48/6/12, `get_dataloaders()` verified; **real PyRadiomics extraction executed 2026-09-23** (all 210 cases, real frozen `top16_features.json`)

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
- [x] Step 7 skeleton: sector tokenizers, class token, transformer encoder, per-sector heads (test with dummy tensors) — `src/models/vvit.py` + `src/models/sectors.py`; self-test passed (embed 128, depth 8, heads 8 → head_dim 16; plan's "head dim 64" incompatible with embed/heads, documented as report deviation)
- [x] Step 9 scaffolding: metrics + bootstrap CI, McNemar, Mann–Whitney wrappers — `src/training/evaluation.py`; self-test passed (AUROC matches sklearn)
- [x] DeLong test implementation *(developed on synthetic predictions)* — `src/stats_tests.py`; self-test passed (AUC cross-check vs sklearn/eval, self-comparison p=1)

### M2: vViT Sanity Check (needs A's M1 pilot data)
- [x] Step 7b: Majority-voting fusion head — `src/models/fusion_baseline_vote.py` (strict majority ≥4/6; 3–3 ties broken by class token); self-test passed
- [x] Step 7c: BCE multi-sector loss + Adam (paper hyperparams: β1=0.9, β2=0.999, ε=1e-8, wd=0) — `src/training/train.py`; self-test passed
- [x] **M2 checkpoint:** model overfits a tiny subset — 16-slice train subset, 60 epochs, loss 0.52→0.0014 (ratio 0.003), vote-proxy acc 1.0
- [x] Integration: vViT + voting + BCE on **real kits19** sector-dict batch (train 48 / val 6 / test 12) — forward `(B,6)` OK; dataset self-test re-run OK
- Note: real pipeline re-run 2026-09-23 — Steps 2–5 on kits19 (210 with segs); **real PyRadiomics extraction executed 2026-09-23** (conda env `radiomics`, all 210 cases, frozen real top-16)

### M3: Full Training
- [x] Step 7d: Full 200-epoch run, best-val checkpoint, TensorBoard logging ✅ (2026-09-23) — `src/training/run_training.py`; best val_acc 0.9277 @ epoch 67; train loss →~0 by ~ep15, val loss climbs (overfit on small set — best-val checkpoint used); artifacts `results/checkpoints/vvit_best.pt`, `results/metrics/train_metrics.jsonl`
- [x] Step 9a: Full metrics + patient-level aggregation ✅ (2026-09-23) — `src/training/run_evaluation.py` on best-val ckpt: test image-level acc 0.531 [0.508–0.551], κ 0.061, AUROC 0.568; patient-level acc 0.500, AUROC 0.639 — **generalization gap vs val_acc 0.93** (overfit; discuss epoch/arch sweep + CV before M5/M6)
- [x] **M3 checkpoint:** paper-style metrics table ✅ (2026-09-23) — `src/training/metrics_table.py` → `results/metrics/table2_style.{md,csv}` (per-sector + fusion, bootstrap CIs, image + patient level); near-chance performance documented (overfit)

### M4: Baselines & Statistical Tests
- [~] Step 8: timm ViT / ConvNeXt / ResNeXt (2D image-only), same protocol — **code + smoke done (2026-09-23)**: `src/training/baselines.py` rewritten as `TimmBaseline` (`vit_small_patch16_224` / `convnext_small` / `resnext50_32x4d`, ImageNet-pretrained + `in_chans=1` + 128→224 resize — pretraining deviation documented); `src/training/run_baselines.py` = Step 7d/9a protocol clone (same splits/aug/Adam, best-val ckpt, test metrics JSON+CSV for Table 3/DeLong input); smoke (2 ep capped) passed all 3 backbones on real kits19. **Full runs pending GPU time** — one command: `python src/training/run_baselines.py`
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
