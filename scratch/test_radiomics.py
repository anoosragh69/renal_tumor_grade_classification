"""
Test script for Step 3: radiomics_extraction.py
Generates fake kits.json metadata for case_00000 and runs the extraction.
"""
import os
import json
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
kits19_data_dir = os.path.join(REPO_ROOT, "kits19", "data")
os.makedirs(kits19_data_dir, exist_ok=True)

# Write fake kits.json if not present
kits_json_path = os.path.join(kits19_data_dir, "kits.json")
if not os.path.exists(kits_json_path):
    fake_records = [
        {"case_id": "case_00000", "age": 55, "sex": "male", "isup_grade": 3}
    ]
    with open(kits_json_path, "w") as f:
        json.dump(fake_records, f, indent=2)
    print(f"Created fake kits.json at {kits_json_path}")
else:
    print(f"kits.json already exists at {kits_json_path}")

# Run extraction
extraction_script = os.path.join(REPO_ROOT, "src", "data", "radiomics_extraction.py")
print("Running radiomics_extraction.py...")
sys.path.insert(0, REPO_ROOT)
os.system(f"python {extraction_script}")
