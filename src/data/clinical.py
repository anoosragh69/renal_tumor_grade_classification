"""
Task 7: Clinical Data Preprocessing for KiTS19 Dataset
"""
import json
import os
import numpy as np
from collections import Counter, defaultdict

DATA_DIR = "/mnt/SharedDrive/Projects/renal_tumor_grade_classification/kits19/data"

NUMERICAL = ["age_at_nephrectomy", "body_mass_index", "radiographic_size", "pathologic_size"]
CATEGORICAL = ["gender", "tumor_histologic_subtype", "smoking_history", "surgery_type", "pathology_t_stage"]
BOOLEANS = ["malignant", "tumor_necrosis"]
COMORBIDITIES = ["myocardial_infarction", "congestive_heart_failure", "copd",
                 "uncomplicated_diabetes_mellitus", "diabetes_mellitus_with_end_organ_damage", "chronic_kidney_disease"]


def load_data(path=None):
    if path is None:
        path = os.path.join(DATA_DIR, "kits.json")
    with open(path, "r") as f:
        return json.load(f)


def extract(cases):
    nums, cats, bools, corbs, labels, ids = [], [], [], [], [], []
    for c in cases:
        isup = c.get("tumor_isup_grade")
        if isup is None:
            continue
        ids.append(c["case_id"])
        nums.append([c.get(f, 0.0) or 0.0 for f in NUMERICAL])
        cats.append([c.get(f, "unknown") or "unknown" for f in CATEGORICAL])
        bools.append([1 if c.get(f, False) else 0 for f in BOOLEANS])
        co = c.get("comorbidities", {})
        corbs.append([1 if co.get(f, False) else 0 for f in COMORBIDITIES])
        labels.append(isup - 1)
    return np.array(nums), cats, np.array(bools), np.array(corbs), np.array(labels), ids


def encode_cats(cats):
    cats_dict = defaultdict(list)
    for row in cats:
        for i, v in enumerate(row):
            if v not in cats_dict[i]:
                cats_dict[i].append(v)
    enc = []
    for row in cats:
        oh = []
        for i, v in enumerate(row):
            for cv in cats_dict[i]:
                oh.append(1 if v == cv else 0)
        enc.append(oh)
    return np.array(enc), dict(cats_dict)


def zscore(arr):
    m = np.mean(arr, axis=0)
    s = np.std(arr, axis=0)
    s[s == 0] = 1
    return (arr - m) / s, m, s


def split(x, y, ids, seed=42):
    n = len(x)
    idx = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(idx)
    t1 = int(n * 0.7)
    t2 = int(n * 0.85)
    return {
        "train": (x[idx[:t1]], y[idx[:t1]], [ids[i] for i in idx[:t1]]),
        "val": (x[idx[t1:t2]], y[idx[t1:t2]], [ids[i] for i in idx[t1:t2]]),
        "test": (x[idx[t2:]], y[idx[t2:]], [ids[i] for i in idx[t2:]]),
    }


def preprocess(json_path=None):
    cases = load_data(json_path)
    xn, xc, xb, xco, y, ids = extract(cases)
    xn_norm, mn, sd = zscore(xn)
    xc_enc, cats = encode_cats(xc)

    cat_names = []
    for i, fn in enumerate(CATEGORICAL):
        for cv in cats[i]:
            cat_names.append(f"{fn}_{cv}")

    all_names = NUMERICAL + cat_names + BOOLEANS + COMORBIDITIES
    x = np.hstack([xn_norm, xc_enc, xb, xco])
    sp = split(x, y, ids)

    return {
        "X": x, "y": y, "feature_names": all_names, "splits": sp,
        "num_mean": mn, "num_std": sd, "categories": cats,
        "num_classes": len(np.unique(y)),
        "class_names": [f"Grade {i+1}" for i in range(len(np.unique(y)))],
    }


if __name__ == "__main__":
    print("Preprocessing clinical data...")
    r = preprocess()
    print(f"Feature matrix: {r['X'].shape}")
    print(f"Target vector: {r['y'].shape}")
    print(f"Features: {len(r['feature_names'])}")
    print(f"Classes: {r['num_classes']} -> {r['class_names']}")
    for name, (xs, ys, _) in r["splits"].items():
        print(f"  {name}: {len(ys)} samples, dist={dict(Counter(ys))}")
    print("Done!")
