"""
Step 4: Synthetic Clinical Fields
==================================
Reads:  kits19/data/kits.json  — real age, sex (and BMI where present)
Writes: data/processed/clinical.csv  — one row per patient_id

Column layout mirrors the vViT sector schema (plan.md §4 Step 6):

  Sector         | Features                           | Source
  ---------------|------------------------------------|-------
  demographic    | age (norm), sex (0/1), bmi (norm)  | real kits.json + synthetic BMI
  comorbidity    | pvd, dm, ckd                       | SYNTHETIC (Bernoulli, Table 1)
  habit          | smoking, alcohol                   | SYNTHETIC (Bernoulli, Table 1)
  label          | binary_label (0=low, 1=high)       | real kits.json isup_grade

A `synthetic` column flags which fields are synthesized.

Paper Table 1 prevalences used for synthesis (label-independent):
  PVD      : 13.5%    (peripheral vascular disease)
  DM       : 24.5%    (diabetes mellitus)
  CKD      : 17.8%    (chronic kidney disease)
  Smoking  : 48.3%    (ever-smoker)
  Alcohol  : 31.7%    (regular alcohol use)
  BMI mean : 30.7, SD : 6.2  (when not in kits.json)

Usage:
  python src/data/synthetic_clinical.py
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paper Table 1 prevalences (label-independent synthesis) ───────────────────
PREVALENCES = {
    "pvd":      0.135,   # peripheral vascular disease
    "dm":       0.245,   # diabetes mellitus
    "ckd":      0.178,   # chronic kidney disease
    "smoking":  0.483,   # ever-smoker
    "alcohol":  0.317,   # regular alcohol use
}
BMI_MEAN = 30.7
BMI_STD  = 6.2

SEED = 42   # versioned seed — document in report


def isup_to_binary(grade):
    """1–2 → 0 (low), 3–4 → 1 (high), None if missing."""
    try:
        g = int(grade)
        return 0 if g <= 2 else 1
    except (TypeError, ValueError):
        return None


def load_kits_json(path):
    if not os.path.exists(path):
        log.error("kits.json not found at %s", path)
        return []
    with open(path) as f:
        return json.load(f)


def build_clinical_df(records, rng):
    """
    Build one row per patient.
    Synthesised fields are drawn independently of the binary label.
    """
    rows = []
    for r in records:
        patient_id = r.get("case_id") or r.get("patient_id")
        if not patient_id:
            continue

        # ── Real fields ───────────────────────────────────────────────────────
        # age: kits.json uses "age_at_nephrectomy" in the full TCIA version,
        # but the GitHub mirror may use "age". Try both.
        age = r.get("age_at_nephrectomy") or r.get("age")
        try:
            age = float(age)
        except (TypeError, ValueError):
            age = float(rng.integers(40, 75))   # impute if missing

        sex_raw = r.get("gender") or r.get("sex") or ""
        sex = 1 if str(sex_raw).lower() in ("male", "m", "1") else 0

        bmi = r.get("body_mass_index") or r.get("bmi")
        bmi_synthetic = False
        try:
            bmi = float(bmi)
            if bmi <= 0:
                raise ValueError
        except (TypeError, ValueError):
            bmi = float(np.clip(rng.normal(BMI_MEAN, BMI_STD), 15.0, 60.0))
            bmi_synthetic = True

        # ── Synthetic comorbidity / habit fields ──────────────────────────────
        pvd     = int(rng.random() < PREVALENCES["pvd"])
        dm      = int(rng.random() < PREVALENCES["dm"])
        ckd     = int(rng.random() < PREVALENCES["ckd"])
        smoking = int(rng.random() < PREVALENCES["smoking"])
        alcohol = int(rng.random() < PREVALENCES["alcohol"])

        # ── Label ─────────────────────────────────────────────────────────────
        isup_raw = r.get("tumor_isup_grade") or r.get("isup_grade")
        label    = isup_to_binary(isup_raw)

        rows.append({
            "patient_id":   patient_id,
            # demographic sector
            "age":          age,
            "sex":          sex,
            "bmi":          bmi,
            # comorbidity sector
            "pvd":          pvd,
            "dm":           dm,
            "ckd":          ckd,
            # habit sector
            "smoking":      smoking,
            "alcohol":      alcohol,
            # metadata
            "label":        label,
            "bmi_synthetic": bmi_synthetic,
            # mark all comorbidity+habit as synthetic
            "synthetic":    True,
        })

    return pd.DataFrame(rows)


def normalize_continuous(df, cols, scaler=None):
    """Z-score normalize continuous columns. Fit if scaler is None."""
    if scaler is None:
        scaler = StandardScaler()
        df[cols] = scaler.fit_transform(df[cols].values)
    else:
        df[cols] = scaler.transform(df[cols].values)
    return df, scaler


def main():
    kits_json_path = os.path.join(REPO_ROOT, "kits19", "data", "kits.json")
    processed_dir  = os.path.join(REPO_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    records = load_kits_json(kits_json_path)
    if not records:
        log.error("No records found in kits.json — aborting.")
        return

    rng = np.random.default_rng(seed=SEED)
    df  = build_clinical_df(records, rng)

    labeled = df["label"].notna().sum()
    log.info(
        "Built clinical DataFrame: %d patients (%d labeled, %d unlabeled)",
        len(df), labeled, len(df) - labeled,
    )
    log.info(
        "Label distribution (labeled): low=%d, high=%d",
        (df["label"] == 0).sum(), (df["label"] == 1).sum(),
    )
    log.info(
        "Synthetic prevalences — pvd=%.1f%% dm=%.1f%% ckd=%.1f%% smoking=%.1f%% alcohol=%.1f%%",
        df["pvd"].mean()*100, df["dm"].mean()*100, df["ckd"].mean()*100,
        df["smoking"].mean()*100, df["alcohol"].mean()*100,
    )

    # Z-score normalize continuous demographic features
    continuous_cols = ["age", "bmi"]
    df, scaler = normalize_continuous(df, continuous_cols)

    # Save
    out_csv = os.path.join(processed_dir, "clinical.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved clinical data → %s", out_csv)

    scaler_path = os.path.join(processed_dir, "clinical_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Saved demographic scaler → %s", scaler_path)

    log.info("Step 4 complete.")
    return df


if __name__ == "__main__":
    main()
