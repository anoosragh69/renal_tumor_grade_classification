"""
Step 5: Exclusion criteria + patient-level stratified splits (plan §3.2–§3.3)
=============================================================================
Exclusion order, per plan §3.2:

  1. Missing any required clinical field      (real kits.json values, Step 4)
  2. Missing WHO/ISUP grade                   (paper: 38 patients)
  3. Image count outside mean ± 1 SD          (paper: 15 patients)
  4. Radiomics extraction failure
  5. No tumour slice with mask >= 256 px

Image count decision (plan §3.5-style, declared in LOG.md 2026-10-24): the paper
does not say what an "image" is. Counting tumour-containing slices excludes 46
patients and counting CT-series slices excludes 1, where the paper excludes 15;
no reading reproduces 15. We use the **CT-series** slice count (`imaging.nii.gz`
z size), the literal reading of "images per patient", which keeps the cohort at
~171 and inside the plan's 157–172 target. Tumour-slice counts are still written
to the split record for transparency.

No class balancing happens here (plan §3.3): balancing is done by subsampling
images **inside the training split only** (src/data/dataset.py). Validation and
test keep the true patient prevalence.

Outputs (data/processed/):
  splits.json              — paper-style split, seed 42 (the dataset default)
  splits_seed43.json       — extra seeds for variance
  splits_seed44.json
  cv_folds.json            — 5-fold patient-level stratified CV over the cohort
  ct_slice_counts.json     — CT-series slice count per case (cached)
  tumor_slice_counts_ge256.json — tumour slices >= 256 px per case (cached)

Usage:
  python src/data/dataset_split.py [--seeds 42 43 44] [--val-frac 0.09] [--test-frac 0.18]
"""

import os
import json
import argparse
import logging
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIN_MASK_PIXELS = 256
CLINICAL_FIELDS = ["age", "sex", "bmi", "pvd", "dm", "ckd", "smoking", "alcohol"]


def get_paths():
    return {
        "kits_json":     os.path.join(REPO_ROOT, "kits19", "data", "kits.json"),
        "data_dir":      os.path.join(REPO_ROOT, "kits19", "data"),
        "interim":       os.path.join(REPO_ROOT, "data", "interim"),
        "clinical":      os.path.join(REPO_ROOT, "data", "processed", "clinical.csv"),
        "radiomics_all": os.path.join(REPO_ROOT, "data", "processed", "radiomics_all.csv"),
        "processed":     os.path.join(REPO_ROOT, "data", "processed"),
    }


def isup_to_binary(grade):
    try:
        g = int(grade)
    except (TypeError, ValueError):
        return None
    if g <= 0:
        return None
    return 0 if g <= 2 else 1


def load_kits_json(path):
    with open(path) as f:
        return {(r.get("case_id") or r.get("patient_id")): r for r in json.load(f)}


def ct_slice_counts(case_ids, data_dir, cache_path):
    """Number of images in each patient's CT series, from the NIfTI header."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            counts = json.load(f)
        log.info("Loaded CT slice counts from %s", cache_path)
    else:
        import SimpleITK as sitk
        counts = {}
        for cid in case_ids:
            img = os.path.join(data_dir, cid, "imaging.nii.gz")
            if not os.path.isfile(img):
                continue
            reader = sitk.ImageFileReader()
            reader.SetFileName(img)
            reader.ReadImageInformation()
            size = reader.GetSize()
            counts[cid] = int(size[2]) if len(size) >= 3 else int(size[-1])
        with open(cache_path, "w") as f:
            json.dump(counts, f, indent=2)
        log.info("Counted CT-series slices for %d cases -> %s", len(counts), cache_path)
    return counts


def tumor_slice_counts(case_ids, data_dir, cache_path):
    """Tumour slices with mask >= MIN_MASK_PIXELS, from the segmentation."""
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            counts = json.load(f)
        log.info("Loaded tumour slice counts from %s", cache_path)
    else:
        import SimpleITK as sitk
        counts = {}
        for cid in case_ids:
            seg = os.path.join(data_dir, cid, "segmentation.nii.gz")
            if not os.path.isfile(seg):
                counts[cid] = 0
                continue
            arr = sitk.GetArrayFromImage(sitk.ReadImage(seg))
            per_slice = (arr > 0).sum(axis=(1, 2))
            counts[cid] = int((per_slice >= MIN_MASK_PIXELS).sum())
        with open(cache_path, "w") as f:
            json.dump(counts, f, indent=2)
        log.info("Counted tumour slices (>= %d px) for %d cases -> %s",
                 MIN_MASK_PIXELS, len(counts), cache_path)
    return counts


def apply_exclusions(kits_meta, clinical_df, radiomics_df, ct_counts, tumor_counts):
    """Return (kept_df, exclusion_log). kept_df: patient_id, label, n_ct, n_tumor."""
    exclusion_log = {
        "missing_clinical":    [],
        "no_isup_grade":       [],
        "image_count_outlier": [],
        "radiomics_failure":   [],
        "no_valid_slices":     [],
    }

    clinical_ok = set(
        clinical_df.dropna(subset=CLINICAL_FIELDS)["patient_id"]
    ) if clinical_df is not None else set(kits_meta)

    # 1+2: clinical fields and grade present
    kept = []
    for pid, meta in kits_meta.items():
        if pid not in clinical_ok:
            exclusion_log["missing_clinical"].append(pid)
            continue
        label = isup_to_binary(meta.get("tumor_isup_grade") or meta.get("isup_grade"))
        if label is None:
            exclusion_log["no_isup_grade"].append(pid)
            continue
        kept.append({"patient_id": pid, "label": label,
                     "n_ct": ct_counts.get(pid, 0), "n_tumor": tumor_counts.get(pid, 0)})
    df = pd.DataFrame(kept)
    log.info("After clinical + grade filter: %d patients (low %d / high %d)",
             len(df), int((df["label"] == 0).sum()), int((df["label"] == 1).sum()))

    # 3: image count beyond mean ± 1 SD (CT-series images, see module docstring)
    mu, sd = df["n_ct"].mean(), df["n_ct"].std()
    lo, hi = mu - sd, mu + sd
    outliers = df[(df["n_ct"] < lo) | (df["n_ct"] > hi)]["patient_id"].tolist()
    exclusion_log["image_count_outlier"].extend(outliers)
    df = df[(df["n_ct"] >= lo) & (df["n_ct"] <= hi)].reset_index(drop=True)
    log.info("After image-count filter (CT slices %.1f ± %.1f = [%.1f, %.1f]): %d patients",
             mu, sd, lo, hi, len(df))

    # 4: radiomics extraction succeeded
    if radiomics_df is not None and not radiomics_df.empty:
        have = set(radiomics_df["patient_id"].unique())
        failed = df[~df["patient_id"].isin(have)]["patient_id"].tolist()
        exclusion_log["radiomics_failure"].extend(failed)
        df = df[df["patient_id"].isin(have)].reset_index(drop=True)
        log.info("After radiomics filter: %d patients", len(df))
    else:
        log.warning("radiomics_all.csv missing — skipping radiomics exclusion filter.")

    # 5: at least one tumour slice with mask >= 256 px
    no_slices = df[df["n_tumor"] == 0]["patient_id"].tolist()
    exclusion_log["no_valid_slices"].extend(no_slices)
    df = df[df["n_tumor"] > 0].reset_index(drop=True)
    log.info("After tumour-slice filter: %d patients", len(df))
    return df, exclusion_log


def stratified_split(df, val_frac, test_frac, seed):
    """Patient-level stratified split; returns sorted id lists."""
    patients, labels = df["patient_id"].tolist(), df["label"].tolist()
    p_trainval, p_test, l_trainval, _ = train_test_split(
        patients, labels, test_size=test_frac, stratify=labels, random_state=seed)
    p_train, p_val, _, _ = train_test_split(
        p_trainval, l_trainval,
        test_size=val_frac / (1 - test_frac),
        stratify=l_trainval, random_state=seed)
    return {"train": sorted(p_train), "val": sorted(p_val), "test": sorted(p_test)}


def build_split_record(df, splits, seed, exclusion_log, ct_mu, ct_sd):
    label_map = dict(zip(df["patient_id"], df["label"]))
    n_ct_map = dict(zip(df["patient_id"], df["n_ct"]))
    n_tumor_map = dict(zip(df["patient_id"], df["n_tumor"]))
    return {
        "seed": seed,
        "train": splits["train"], "val": splits["val"], "test": splits["test"],
        "counts": {k: len(splits[k]) for k in ("train", "val", "test")},
        "counts_total": len(df),
        "label_distribution": {
            name: {"low": sum(1 for p in ids if label_map.get(p) == 0),
                   "high": sum(1 for p in ids if label_map.get(p) == 1)}
            for name, ids in [("train", splits["train"]), ("val", splits["val"]),
                              ("test", splits["test"])]
        },
        "image_count_rule": {
            "metric": "CT-series slice count",
            "mean": float(ct_mu), "std": float(ct_sd),
            "note": ("tumour-slice counts exclude 46 patients where the paper excludes 15; "
                     "CT-series counts are the literal reading of images per patient"),
        },
        "tumor_slice_counts": {p: n_tumor_map[p] for p in df["patient_id"]},
        "excluded": {k: len(v) for k, v in exclusion_log.items()},
        "excluded_ids": exclusion_log,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 5: exclusions + stratified splits")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--val-frac", type=float, default=0.09, help="≈15/172")
    parser.add_argument("--test-frac", type=float, default=0.18, help="≈30/172")
    args = parser.parse_args()
    paths = get_paths()

    kits_meta = load_kits_json(paths["kits_json"])
    clinical_df = (pd.read_csv(paths["clinical"])
                   if os.path.exists(paths["clinical"]) else None)
    radiomics_df = (pd.read_csv(paths["radiomics_all"], usecols=["patient_id"])
                    if os.path.exists(paths["radiomics_all"]) else None)

    ct_counts = ct_slice_counts(sorted(kits_meta), paths["data_dir"],
                                os.path.join(paths["processed"], "ct_slice_counts.json"))
    tumor_counts = tumor_slice_counts(sorted(kits_meta), paths["data_dir"],
                                      os.path.join(paths["processed"],
                                                   "tumor_slice_counts_ge256.json"))

    df, exclusion_log = apply_exclusions(
        kits_meta, clinical_df, radiomics_df, ct_counts, tumor_counts)
    log.info("Final cohort: %d patients (low %d / high %d)",
             len(df), int((df["label"] == 0).sum()), int((df["label"] == 1).sum()))

    ct_mu, ct_sd = df["n_ct"].mean(), df["n_ct"].std()

    # Paper-style split, one file per seed; seed 42 is the dataset default.
    for seed in args.seeds:
        record = build_split_record(
            df, stratified_split(df, args.val_frac, args.test_frac, seed),
            seed, exclusion_log, ct_mu, ct_sd)
        name = "splits.json" if seed == args.seeds[0] else f"splits_seed{seed}.json"
        path = os.path.join(paths["processed"], name)
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        log.info("seed %d -> %s : train %d / val %d / test %d (low/high %s)",
                 seed, name, record["counts"]["train"], record["counts"]["val"],
                 record["counts"]["test"], record["label_distribution"])

    # 5-fold patient-level stratified CV over the whole cohort
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seeds[0])
    ids, labels = df["patient_id"].tolist(), df["label"].tolist()
    cv_folds = {
        f"fold_{i}": {"train": [ids[a] for a in tr], "val": [ids[b] for b in va]}
        for i, (tr, va) in enumerate(skf.split(ids, labels))
    }
    cv_path = os.path.join(paths["processed"], "cv_folds.json")
    with open(cv_path, "w") as f:
        json.dump(cv_folds, f, indent=2)
    log.info("Saved 5-fold CV -> %s (fold val sizes: %s)",
             cv_path, [len(v["val"]) for v in cv_folds.values()])
    log.info("Exclusion summary: %s", exclusion_log)


if __name__ == "__main__":
    main()
