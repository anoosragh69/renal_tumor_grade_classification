"""
Step 4: Clinical Features from kits.json
=========================================
Reads:  kits19/data/kits.json
Writes: data/processed/clinical.csv — one row per patient (clinical values only)

Column layout mirrors the vViT sector schema (plan §3.1):

  Sector         | Columns              | kits.json source
  ---------------|----------------------|----------------------------------------
  demographic    | age, sex, bmi        | age_at_nephrectomy, gender, body_mass_index
  comorbidity    | pvd, dm, ckd         | comorbidities.*
  habit          | smoking, alcohol     | smoking_history, alcohol_use
  label          | label                | tumor_isup_grade (NaN if unlabeled)

No values are synthesized or imputed. Continuous demographic fields are written
raw; train-only z-scoring is applied by the dataset using the train split.

Our encoding choices (the paper does not specify these):
  sex      : male=1, female=0
  PVD      : comorbidities.peripheral_vascular_disease
  DM       : uncomplicated_diabetes_mellitus OR diabetes_mellitus_with_end_organ_damage
  CKD      : comorbidities.chronic_kidney_disease
  smoking  : never=0, previous=1, current=2
  alcohol  : never_or_not_in_last_3mo=0, two_or_less_daily=1, more_than_two_daily=2
  label    : 0=low (ISUP 1-2), 1=high (ISUP 3-4), NaN=no grade

Usage:
  python src/data/clinical.py
"""

import os
import json
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEX_CODES = {"male": 1, "female": 0}
SMOKING_CODES = {"never_smoked": 0, "previous_smoker": 1, "current_smoker": 2}
ALCOHOL_CODES = {
    "never_or_not_in_last_3mo": 0,
    "two_or_less_daily": 1,
    "more_than_two_daily": 2,
}


def isup_to_binary(grade):
    """1–2 → 0 (low), 3–4 → 1 (high), None if missing."""
    try:
        g = int(grade)
    except (TypeError, ValueError):
        return None
    if g <= 0:
        return None
    return 0 if g <= 2 else 1


def load_kits_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"kits.json not found at {path}")
    with open(path) as f:
        return json.load(f)


def _map_code(value, mapping, field, case_id, warnings):
    if value is None or value == "":
        warnings.append(f"{case_id}: missing {field}")
        return np.nan
    key = str(value).strip().lower()
    if key not in mapping:
        warnings.append(f"{case_id}: unmapped {field}={value!r}")
        return np.nan
    return mapping[key]


def build_clinical_df(records):
    """Build one row per patient from real kits.json values only."""
    rows = []
    warnings = []
    for r in records:
        patient_id = r.get("case_id") or r.get("patient_id")
        if not patient_id:
            warnings.append("record without case_id — skipped")
            continue

        co = r.get("comorbidities") or {}
        dm = int(bool(co.get("uncomplicated_diabetes_mellitus"))) or int(
            bool(co.get("diabetes_mellitus_with_end_organ_damage"))
        )

        rows.append({
            "patient_id": patient_id,
            # demographic sector
            "age": r.get("age_at_nephrectomy"),
            "sex": _map_code(r.get("gender"), SEX_CODES, "gender", patient_id, warnings),
            "bmi": r.get("body_mass_index"),
            # comorbidity sector
            "pvd": int(bool(co.get("peripheral_vascular_disease"))),
            "dm": dm,
            "ckd": int(bool(co.get("chronic_kidney_disease"))),
            # habit sector
            "smoking": _map_code(r.get("smoking_history"), SMOKING_CODES,
                                 "smoking_history", patient_id, warnings),
            "alcohol": _map_code(r.get("alcohol_use"), ALCOHOL_CODES,
                                 "alcohol_use", patient_id, warnings),
            # target
            "label": isup_to_binary(r.get("tumor_isup_grade")),
        })

    for msg in warnings:
        log.warning(msg)
    return pd.DataFrame(rows)


def validate(df):
    """Fail loudly if any model-facing clinical field is missing."""
    required = ["age", "sex", "bmi", "pvd", "dm", "ckd", "smoking", "alcohol"]
    missing = df[required].isna().sum()
    if missing.any():
        raise ValueError(f"Missing clinical values:\n{missing[missing > 0]}")
    if not df["sex"].isin([0, 1]).all():
        raise ValueError("sex must be 0/1")
    if not df["smoking"].isin([0, 1, 2]).all():
        raise ValueError("smoking must be 0/1/2")
    if not df["alcohol"].isin([0, 1, 2]).all():
        raise ValueError("alcohol must be 0/1/2")
    if df["age"].min() <= 0 or df["bmi"].min() <= 0:
        raise ValueError("age and bmi must be positive")
    suspicious = df[df["age"] < 18]["patient_id"].tolist()
    if suspicious:
        log.warning("Patients with age < 18 (check source data): %s", suspicious)


def main():
    kits_json_path = os.path.join(REPO_ROOT, "kits19", "data", "kits.json")
    processed_dir = os.path.join(REPO_ROOT, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    records = load_kits_json(kits_json_path)
    df = build_clinical_df(records)
    validate(df)

    labeled = df["label"].notna()
    log.info("Built clinical table: %d patients (%d labeled, %d unlabeled)",
             len(df), int(labeled.sum()), int((~labeled).sum()))
    log.info("Label distribution (labeled): low=%d, high=%d",
             int((df.loc[labeled, "label"] == 0).sum()),
             int((df.loc[labeled, "label"] == 1).sum()))
    log.info("Prevalence: pvd=%.1f%% dm=%.1f%% ckd=%.1f%% ever_smoker=%.1f%% any_alcohol=%.1f%%",
             df["pvd"].mean() * 100, df["dm"].mean() * 100, df["ckd"].mean() * 100,
             (df["smoking"] > 0).mean() * 100, (df["alcohol"] > 0).mean() * 100)

    out_csv = os.path.join(processed_dir, "clinical.csv")
    df.to_csv(out_csv, index=False)
    log.info("Saved clinical data → %s", out_csv)
    return df


if __name__ == "__main__":
    main()
