"""
M1 pilot: generate a ~40-case MOCK KiTS19-style dataset (seed=42).
================================================______________________
Purpose: exercise Steps 2-6 end-to-end (including all Step 5 exclusion
paths) while the real KiTS19 path is pending. kits19/* is gitignored.

Writes:
  kits19/data/case_00000..case_00039/imaging.nii.gz + segmentation.nii.gz
  kits19/data/kits.json          (overwrites the 1-record test fixture)

Design (all draws from np.random.default_rng(42) - versioned):
  - 40 cases, CT noise N(0,100) HU at 512x512, 6-30 slices
  - 4 cases with z in {6,6,24,30}  -> slice-count outlier exclusion
  - 4 cases with tumor_isup_grade = null -> no-grade exclusion
  - remaining 32 labelled ~65% low (ISUP 1-2) / 35% high (ISUP 3-4),
    slice counts z in [10,14]
  - tumour: contiguous z block, random x/y box (>=600 px per slice)
  - key names mirror the real GitHub mirror: tumor_isup_grade,
    age_at_nephrectomy, gender (body_mass_index omitted -> synthetic BMI)

Expected after exclusions + equalise (plan S5): ~20-24 patients.
"""
import os
import json
import numpy as np
import SimpleITK as sitk

SEED = 42
N_CASES = 40
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "kits19", "data")

# indices fixed before any randomness so exclusions are deterministic:
# 4 slice-count extremes, 4 null grades (chosen from normal-z cases only)
EXTREME_IDX = {5: 6, 11: 6, 23: 24, 31: 30}          # case_idx -> z slices
NULL_GRADE_IDX = {3, 17, 28, 36}


def make_case(rng, z_slices, isup_grade, case_id):
    # CT noise in HU
    img = np.random.default_rng(int(rng.integers(0, 2**31))).normal(0, 100, (z_slices, 512, 512)).astype(np.int16)

    # tumour: contiguous z block centered in volume, random x/y box
    t_z = max(2, min(z_slices, int(rng.integers(2, min(6, z_slices) + 1))))
    z0 = max(0, (z_slices - t_z) // 2)
    cy = int(rng.integers(150, 362))
    cx = int(rng.integers(150, 362))
    dy = int(rng.integers(12, 31))   # half-height 12..30  -> box 24..60 rows
    dx = int(rng.integers(15, 41))   # half-width  15..40  -> box 30..80 cols

    seg = np.zeros((z_slices, 512, 512), dtype=np.uint8)
    seg[z0:z0 + t_z, cy - dy:cy + dy, cx - dx:cx + dx] = 1

    case_dir = os.path.join(OUT_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)
    for name, arr in (("imaging.nii.gz", img), ("segmentation.nii.gz", seg)):
        sitk_img = sitk.GetImageFromArray(arr)
        sitk_img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(sitk_img, os.path.join(case_dir, name))


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    records = []
    for i in range(N_CASES):
        case_id = f"case_{i:05d}"
        z_slices = EXTREME_IDX.get(i, int(rng.integers(10, 15)))

        # labels: ~65/35 low/high among all non-null cases
        if i in NULL_GRADE_IDX:
            isup = None
        else:
            isup = int(rng.choice([1, 2], p=[0.55, 0.45])) if rng.random() < 0.65 \
                else int(rng.choice([3, 4], p=[0.7, 0.3]))

        make_case(rng, z_slices, isup, case_id)
        records.append({
            "case_id": case_id,
            "age_at_nephrectomy": int(rng.integers(40, 80)),
            "gender": str(rng.choice(["male", "female"])),
            "tumor_isup_grade": isup,
            # body_mass_index intentionally absent -> Step 4 synthetic path
        })

    kits_json = os.path.join(OUT_DIR, "kits.json")
    with open(kits_json, "w") as f:
        json.dump(records, f, indent=2)

    n_null = sum(1 for r in records if r["tumor_isup_grade"] is None)
    n_low = sum(1 for r in records if (r["tumor_isup_grade"] or 0) <= 2)
    n_high = sum(1 for r in records if (r["tumor_isup_grade"] or 0) >= 3)
    print(f"Generated {N_CASES} mock cases in {OUT_DIR}")
    print(f"kits.json: {n_low} low / {n_high} high / {n_null} null-grade")
    print(f"slice-count outliers at z={sorted(set(EXTREME_IDX.values()))} "
          f"(indices {sorted(EXTREME_IDX)}), null grades at {sorted(NULL_GRADE_IDX)}")


if __name__ == "__main__":
    main()
