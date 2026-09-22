"""Task 5: Data Exploration & Analysis for KiTS19 Dataset"""
import json
import os
import statistics
from collections import Counter

DATA_DIR = "/mnt/SharedDrive/Projects/renal_tumor_grade_classification/kits19/data"

# Load clinical metadata
with open(os.path.join(DATA_DIR, "kits.json"), "r") as f:
    cases = json.load(f)

print(f"Total cases in metadata: {len(cases)}")

# Check how many have imaging
case_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith("case_")]
print(f"Cases with imaging directories: {len(case_dirs)}")

# --- ISUP Grade Distribution ---
isup_grades = [c["tumor_isup_grade"] for c in cases]
null_count = sum(1 for g in isup_grades if g is None)
valid_grades = [g for g in isup_grades if g is not None]

print(f"\n=== ISUP Grade Distribution ===")
print(f"Cases with ISUP grade: {len(valid_grades)}")
print(f"Cases with null ISUP grade (benign): {null_count}")

grade_counts = Counter(valid_grades)
for grade in sorted(grade_counts.keys()):
    pct = grade_counts[grade] / len(valid_grades) * 100
    print(f"  Grade {grade}: {grade_counts[grade]} cases ({pct:.1f}%)")

max_count = max(grade_counts.values())
min_count = min(grade_counts.values())
print(f"\nImbalance ratio (max/min): {max_count/min_count:.2f}")

# --- Missing Values Analysis ---
print(f"\n=== Missing Values Analysis ===")
key_features = [
    "age_at_nephrectomy", "gender", "body_mass_index",
    "tumor_isup_grade", "radiographic_size", "pathologic_size",
    "tumor_histologic_subtype", "malignant", "smoking_history",
    "surgery_type", "pathology_t_stage"
]

for feat in key_features:
    missing = sum(1 for c in cases if c.get(feat) is None)
    if missing > 0:
        print(f"  {feat}: {missing} missing ({missing/len(cases)*100:.1f}%)")
    else:
        print(f"  {feat}: 0 missing (complete)")

# --- Numerical Feature Statistics ---
print(f"\n=== Numerical Feature Statistics ===")
numerical_features = ["age_at_nephrectomy", "body_mass_index", "radiographic_size", "pathologic_size"]
for feat in numerical_features:
    vals = [c[feat] for c in cases if c.get(feat) is not None]
    if vals:
        mn = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {feat}: mean={mn:.1f}, std={sd:.1f}, min={min(vals):.1f}, max={max(vals):.1f}")

# --- Categorical Feature Distributions ---
print(f"\n=== Categorical Feature Distributions ===")
gender_counts = Counter(c["gender"] for c in cases)
print(f"\nGender:")
for g, cnt in gender_counts.items():
    print(f"  {g}: {cnt} ({cnt/len(cases)*100:.1f}%)")

hist_counts = Counter(c["tumor_histologic_subtype"] for c in cases if c.get("tumor_histologic_subtype"))
print(f"\nTumor histologic subtype:")
for h, cnt in hist_counts.most_common():
    print(f"  {h}: {cnt} ({cnt/len(cases)*100:.1f}%)")

mal_counts = Counter(c["malignant"] for c in cases)
print(f"\nMalignant:")
for m, cnt in mal_counts.items():
    print(f"  {m}: {cnt} ({cnt/len(cases)*100:.1f}%)")

surg_counts = Counter(c["surgery_type"] for c in cases if c.get("surgery_type"))
print(f"\nSurgery type:")
for s, cnt in surg_counts.most_common():
    print(f"  {s}: {cnt} ({cnt/len(cases)*100:.1f}%)")

print("\n=== Task 5 Complete ===")
