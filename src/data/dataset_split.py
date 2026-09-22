"""
Step 5: Exclusion Criteria + Stratified Dataset Split
=======================================================
Implements the paper's Fig. 2 exclusion/split pipeline:

  1. Start with all patients that have a WHO/ISUP grade in kits.json
  2. Exclude: missing any required clinical field  (≈0 after Step 4 synthesis)
  3. Exclude: image count outside mean ± 1 SD of all patients  (plan §5.3)
  4. Exclude: radiomics extraction failures (not in radiomics_all.csv)
  5. Exclude: no 2D tumour slice with mask ≥ 256 px  (plan §5.4)
  6. Within each class (low/high): random-select to equalise image counts
     across classes  (plan §5.5)
  7. Stratified random split → train / val / test  (target: ~111/15/30)

Reads:
  kits19/data/kits.json
  data/processed/radiomics_all.csv
  data/interim/<patient_id>/      (to count valid slices)

Writes:
  data/processed/splits.json  — {train: [...], val: [...], test: [...], seed: int,
                                  excluded: {...}, counts: {...}}

Usage:
  python src/data/dataset_split.py [--seed 42] [--val-frac 0.09] [--test-frac 0.18]
"""

import os
import json
import argparse
import logging
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_MASK_PIXELS = 256


def get_paths():
    return {
        "kits_json":       os.path.join(REPO_ROOT, "kits19", "data", "kits.json"),
        "interim":         os.path.join(REPO_ROOT, "data", "interim"),
        "radiomics_all":   os.path.join(REPO_ROOT, "data", "processed", "radiomics_all.csv"),
        "processed":       os.path.join(REPO_ROOT, "data", "processed"),
    }


def isup_to_binary(grade):
    try:
        g = int(grade)
        return 0 if g <= 2 else 1
    except (TypeError, ValueError):
        return None


def load_kits_json(path):
    with open(path) as f:
        records = json.load(f)
    return {(r.get("case_id") or r.get("patient_id")): r for r in records}


def count_valid_slices(patient_id, interim_dir):
    """Count how many saved patches exist for a patient (each == 1 valid tumour slice)."""
    pat_dir = os.path.join(interim_dir, patient_id)
    return len(glob.glob(os.path.join(pat_dir, "slice_*.npy")))


def apply_exclusions(kits_meta, radiomics_df, interim_dir):
    """
    Returns (kept_df, exclusion_log) where kept_df has columns:
      patient_id, label, n_slices
    """
    exclusion_log = {
        "no_isup_grade":       [],
        "radiomics_failure":   [],
        "no_valid_slices":     [],
        "slice_count_outlier": [],
    }

    # ── Step 1: start with patients that have a label ─────────────────────────
    labeled_patients = []
    for pid, meta in kits_meta.items():
        isup_raw = meta.get("tumor_isup_grade") or meta.get("isup_grade")
        label = isup_to_binary(isup_raw)
        if label is None:
            exclusion_log["no_isup_grade"].append(pid)
            continue
        labeled_patients.append({"patient_id": pid, "label": label})

    df = pd.DataFrame(labeled_patients)
    log.info("Patients with ISUP grade: %d", len(df))

    # ── Step 2: radiomics extraction must have succeeded ──────────────────────
    if radiomics_df is not None and not radiomics_df.empty:
        patients_with_radiomics = set(radiomics_df["patient_id"].unique())
        failed = df[~df["patient_id"].isin(patients_with_radiomics)]["patient_id"].tolist()
        exclusion_log["radiomics_failure"].extend(failed)
        df = df[df["patient_id"].isin(patients_with_radiomics)].reset_index(drop=True)
        log.info("After radiomics filter: %d", len(df))
    else:
        log.warning("radiomics_all.csv not found or empty — skipping radiomics exclusion filter.")

    # ── Step 3: count valid tumour slices ─────────────────────────────────────
    df["n_slices"] = df["patient_id"].apply(lambda pid: count_valid_slices(pid, interim_dir))
    no_slices = df[df["n_slices"] == 0]["patient_id"].tolist()
    exclusion_log["no_valid_slices"].extend(no_slices)
    df = df[df["n_slices"] > 0].reset_index(drop=True)
    log.info("After no-valid-slice filter: %d", len(df))

    # ── Step 4: image count outlier exclusion (mean ± 1 SD) ──────────────────
    if len(df) > 2:
        mu = df["n_slices"].mean()
        sd = df["n_slices"].std()
        lo, hi = mu - sd, mu + sd
        outliers = df[(df["n_slices"] < lo) | (df["n_slices"] > hi)]["patient_id"].tolist()
        exclusion_log["slice_count_outlier"].extend(outliers)
        df = df[(df["n_slices"] >= lo) & (df["n_slices"] <= hi)].reset_index(drop=True)
        log.info(
            "After slice-count outlier filter (%.1f ± %.1f = [%.1f, %.1f]): %d",
            mu, sd, lo, hi, len(df),
        )
    else:
        log.warning("Too few patients for outlier filtering — skipping.")

    return df, exclusion_log


def equalise_class_image_counts(df, rng):
    """
    Plan §5.5: random-select patients within each class so total image counts
    are approximately equal between low and high. We down-sample the class
    with more total slices by removing patients (whole patient, not slices).
    """
    low  = df[df["label"] == 0].copy()
    high = df[df["label"] == 1].copy()

    total_low  = low["n_slices"].sum()
    total_high = high["n_slices"].sum()
    log.info("Before equalisation: low=%d slices (%d pts), high=%d slices (%d pts)",
             total_low, len(low), total_high, len(high))

    # Down-sample the majority class by removing patients greedily
    if total_low > total_high:
        majority, minority = low, high
    else:
        majority, minority = high, low

    # Shuffle majority and drop patients until sums are roughly equal
    majority = majority.sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    target = minority["n_slices"].sum()
    cumsum = 0
    keep_idx = []
    for i, row in majority.iterrows():
        if cumsum + row["n_slices"] <= target * 1.1:   # allow 10% overshoot
            keep_idx.append(i)
            cumsum += row["n_slices"]
    majority = majority.loc[keep_idx].reset_index(drop=True)

    df_eq = pd.concat([minority, majority]).reset_index(drop=True)
    log.info("After equalisation: low=%d pts, high=%d pts (total=%d)",
             (df_eq["label"]==0).sum(), (df_eq["label"]==1).sum(), len(df_eq))
    return df_eq


def stratified_split(df, val_frac, test_frac, seed):
    """
    Stratified split by binary label.
    Returns dict: {train: [ids], val: [ids], test: [ids]}
    """
    patients = df["patient_id"].tolist()
    labels   = df["label"].tolist()

    # First carve out test set
    p_trainval, p_test, l_trainval, _ = train_test_split(
        patients, labels,
        test_size=test_frac,
        stratify=labels,
        random_state=seed,
    )

    # Then val from trainval
    val_frac_adjusted = val_frac / (1 - test_frac)
    p_train, p_val, _, _ = train_test_split(
        p_trainval, l_trainval,
        test_size=val_frac_adjusted,
        stratify=l_trainval,
        random_state=seed,
    )

    return {
        "train": sorted(p_train),
        "val":   sorted(p_val),
        "test":  sorted(p_test),
    }


def main():
    parser = argparse.ArgumentParser(description="Step 5: Exclusions + stratified split")
    parser.add_argument("--seed",      type=int,   default=42,   help="Random seed (versioned)")
    parser.add_argument("--val-frac",  type=float, default=0.09, help="Val fraction of full set (default 0.09 ≈ 15/172)")
    parser.add_argument("--test-frac", type=float, default=0.18, help="Test fraction of full set (default 0.18 ≈ 30/172)")
    args = parser.parse_args()

    paths = get_paths()
    rng   = np.random.default_rng(seed=args.seed)

    # Load data
    if not os.path.exists(paths["kits_json"]):
        log.error("kits.json not found — run generate_fake_data.py or provide real dataset.")
        return

    kits_meta = load_kits_json(paths["kits_json"])
    log.info("Loaded %d records from kits.json", len(kits_meta))

    radiomics_df = None
    if os.path.exists(paths["radiomics_all"]):
        radiomics_df = pd.read_csv(paths["radiomics_all"])

    # Apply exclusions
    df, exclusion_log = apply_exclusions(kits_meta, radiomics_df, paths["interim"])

    if len(df) < 3:
        log.warning(
            "Only %d patients remain after exclusions. "
            "Stratified split requires at least 3. Saving what we have.", len(df)
        )
        splits = {"train": df["patient_id"].tolist(), "val": [], "test": []}
    else:
        # Equalise class image counts (plan §5.5)
        df = equalise_class_image_counts(df, rng)

        # Stratified split
        splits = stratified_split(df, args.val_frac, args.test_frac, args.seed)

    # Compile output
    label_map = dict(zip(df["patient_id"], df["label"]))
    output = {
        "seed":     args.seed,
        "train":    splits["train"],
        "val":      splits["val"],
        "test":     splits["test"],
        "counts": {
            "train": len(splits["train"]),
            "val":   len(splits["val"]),
            "test":  len(splits["test"]),
            "total": len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
        },
        "label_distribution": {
            split: {
                "low":  sum(1 for p in ids if label_map.get(p) == 0),
                "high": sum(1 for p in ids if label_map.get(p) == 1),
            }
            for split, ids in [("train", splits["train"]), ("val", splits["val"]), ("test", splits["test"])]
        },
        "excluded": {k: len(v) for k, v in exclusion_log.items()},
        "excluded_ids": exclusion_log,
    }

    out_path = os.path.join(paths["processed"], "splits.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Saved splits.json → %s", out_path)
    log.info(
        "Split counts: train=%d, val=%d, test=%d",
        output["counts"]["train"], output["counts"]["val"], output["counts"]["test"],
    )
    log.info("Exclusion summary: %s", output["excluded"])
    log.info("Step 5 complete.")


if __name__ == "__main__":
    main()
