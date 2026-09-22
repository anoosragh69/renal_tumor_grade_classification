# WHO/ISUP Grade Availability Audit (Step 1)

**Date:** 2026-09-22
**Source audited:** `kits19/data/kits.json` + on-disk imaging/segmentations (neheller/kits19 GitHub mirror)

## Inventory

| Asset | Count |
|---|---|
| Imaging volumes (`imaging.nii.gz`) | 300 (case_00000–00299) |
| Segmentations (`segmentation.nii.gz`) | 210 |
| `kits.json` case entries | 210 |
| Segmentation IDs == kits.json IDs | yes (exact match) |

Only 210/300 cases carry annotations + metadata; the other 90 have imaging only.

## `tumor_isup_grade` availability

| | Count |
|---|---|
| Total kits.json cases | 210 |
| With grade | **172** |
| Missing grade (`null`) | 38 |
| With grade **and** segmentation | **172** (all labeled cases have masks) |

### Grade distribution (multiclass)

| Grade | N |
|---|---|
| 1 | 26 |
| 2 | 93 |
| 3 | 39 |
| 4 | 14 |
| 5 | 0 |

### Binary low/high (paper's task: low = ISUP 1–2, high = ISUP ≥3)

| Class | N |
|---|---|
| Low | 119 |
| High | 53 |
| **Total usable** | **172** |

Class imbalance ratio ≈ 2.24:1 (low:high).

## Implications for the plan

1. **Usable N = 172** before exclusions (Step 5). The paper reports 157 after exclusions → 111/15/30 split; our 172 is close and plausible to approach after image-count / mask-size criteria.
2. **No ISUP grade 5** in this pull — binary mapping (low ≤2 / high ≥3) is unaffected.
3. **Source deviation:** GitHub kits19 mirror (not TCIA C4KC-KiTS as the paper cites). Grade counts may differ from the paper's; declare in report deviations (plan §6).
4. **90 imaging-only cases** cannot be used (no mask → no crop/radiomics, no metadata → no label). Exclusion criterion in Step 5 is effectively "must be in the 210 annotated set."
5. 38/210 missing grades (~18%) — larger than expected; likely benign/no-grade cases. Excluded per Fig. 2 pipeline.

## Recommendation

Proceed with the GitHub mirror: all 172 labeled cases have segmentations, sufficient for replication with documented source deviation. No TCIA re-pull required unless grade-count fidelity to the paper becomes critical.
