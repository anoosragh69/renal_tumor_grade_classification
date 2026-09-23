"""
Step 3: PyRadiomics Feature Extraction + Top-16 F-value Selection (train-only)
==============================================================================
Reads:
  - data/interim/<patient_id>/slice_<idx>.npy  — cropped 2D image patches (float32, [0,1])
  - kits19/data/<patient_id>/segmentation.nii.gz — original 3D mask (for 2D slice masks)
  - kits.json (via src/data/clinical.py or direct JSON load) — for binary label (low=0, high=1)

Writes:
  - data/processed/radiomics_all.csv   — one row per (patient_id, slice_id), all ~105 features
  - data/processed/radiomics_top16.csv — only top-16 features selected on train split
  - data/processed/top16_features.json — frozen feature list (load this in val/test — no re-selection)
  - data/processed/radiomics_scaler.pkl — StandardScaler fit on train split

Usage:
  python src/data/radiomics_extraction.py [--splits data/processed/splits.json]

  If splits.json doesn't exist yet (Step 5 not done), runs in 'unsplit' mode and
  skips F-value selection — useful for piloting on mock data.
"""
import os
import sys
import json
import glob
import argparse
import logging
import pickle
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── PyRadiomics availability check ────────────────────────────────────────────
# pyradiomics has no cp314 wheels; use the Miniconda env:
#   %USERPROFILE%\miniconda3\envs\radiomics\python.exe
try:
    import radiomics  # noqa: F401
    from radiomics import featureextractor as _fe  # noqa: F401
    PYRADIOMICS_AVAILABLE = True
except ImportError:
    PYRADIOMICS_AVAILABLE = False
    log.warning(
        "pyradiomics not installed. Running in MOCK mode. "
        "Run with the conda env python for real features."
    )

# Cached extractor — rebuild once per process (enableAllFeatures is not free).
# normalize=True z-scores intensities so binWidth=25 is meaningful and ~20×
# faster on uint16-scaled crops (raw binWidth=25 → ~10s/slice → ~0.4s/slice).
_EXTRACTOR = None


def _get_extractor():
    global _EXTRACTOR
    if _EXTRACTOR is None:
        from radiomics import featureextractor
        logging.getLogger("radiomics").setLevel(logging.ERROR)
        _EXTRACTOR = featureextractor.RadiomicsFeatureExtractor(
            force2D=True,
            force2Ddimension=0,
            binWidth=25,
            normalize=True,
            resampledPixelSpacing=None,
            interpolator="sitkBSpline",
            additionalInfo=False,  # skip diagnostics_* keys
        )
        _EXTRACTOR.enableAllFeatures()
    return _EXTRACTOR


def _to_float(v):
    """Coerce pyradiomics values (often 0-d/1-element ndarray) to float; None if not scalar."""
    if isinstance(v, np.ndarray):
        if v.size == 1:
            return float(v.reshape(-1)[0])
        return None
    if isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    return None

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_SIZE = (128, 128)
MIN_MASK_PIXELS = 256   # plan §5.4: exclude if tumour mask < 256 px in 2D slice
TOP_K = 16              # top-K features by ANOVA F-value (train-only)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_paths():
    return {
        "kits19_data": os.path.join(REPO_ROOT, "kits19", "data"),
        "interim":     os.path.join(REPO_ROOT, "data", "interim"),
        "processed":   os.path.join(REPO_ROOT, "data", "processed"),
        "kits_json":   os.path.join(REPO_ROOT, "kits19", "data", "kits.json"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_kits_json(kits_json_path):
    """Return dict keyed by 'case_XXXXX' → {age, sex, isup_grade, ...}."""
    if not os.path.exists(kits_json_path):
        log.warning("kits.json not found at %s — labels unavailable", kits_json_path)
        return {}
    with open(kits_json_path) as f:
        records = json.load(f)
    # kits.json is a list of dicts with 'case_id' keys
    return {r["case_id"]: r for r in records}


def isup_to_binary(grade):
    """Convert numeric ISUP/WHO grade to binary: 1-2 → 0 (low), 3-4 → 1 (high). None if missing."""
    try:
        g = int(grade)
        if 1 <= g <= 2:
            return 0
        elif 3 <= g <= 4:
            return 1
    except (TypeError, ValueError):
        pass
    return None


def extract_2d_radiomics(img_arr_2d, mask_arr_2d, patient_id, slice_idx, mock=False):
    """
    Run PyRadiomics on a single 2D slice.
    img_arr_2d  : float32 [0,1] 128×128 numpy array
    mask_arr_2d : uint8 binary mask 128×128 numpy array
    mock        : if True, return synthetic feature dict (for testing without pyradiomics)
    Returns dict of feature_name → value, or None if extraction fails.
    """
    mask_uint = (mask_arr_2d > 0).astype(np.uint8)
    if mask_uint.sum() < MIN_MASK_PIXELS:
        return None

    if mock or not PYRADIOMICS_AVAILABLE:
        # Return 105 reproducible synthetic features for pipeline testing.
        # Deterministic seed: Python's hash() is salted per-process (PYTHONHASHSEED),
        # which made mock features non-reproducible across runs; crc32 is stable.
        import zlib
        stable_seed = zlib.crc32(f"{patient_id}:{slice_idx}".encode()) % (2**31)
        rng = np.random.default_rng(seed=stable_seed)
        feature_names = (
            [f"original_firstorder_{n}" for n in ["Energy","Entropy","Kurtosis","Maximum","MeanAbsoluteDeviation",
             "Mean","Median","Minimum","Range","RobustMeanAbsoluteDeviation","RootMeanSquared","Skewness",
             "TotalEnergy","Uniformity","Variance"]] +
            [f"original_glcm_{n}" for n in ["Autocorrelation","ClusterProminence","ClusterShade","ClusterTendency",
             "Contrast","Correlation","DifferenceAverage","DifferenceEntropy","DifferenceVariance",
             "Id","Idm","Idmn","Idn","Imc1","Imc2","InverseVariance","JointAverage","JointEnergy",
             "JointEntropy","MCC","MaximumProbability","SumAverage","SumEntropy","SumSquares"]] +
            [f"original_glrlm_{n}" for n in ["GrayLevelNonUniformity","GrayLevelNonUniformityNormalized",
             "GrayLevelVariance","HighGrayLevelRunEmphasis","LongRunEmphasis","LongRunHighGrayLevelEmphasis",
             "LongRunLowGrayLevelEmphasis","LowGrayLevelRunEmphasis","RunEntropy","RunLengthNonUniformity",
             "RunLengthNonUniformityNormalized","RunPercentage","RunVariance","ShortRunEmphasis",
             "ShortRunHighGrayLevelEmphasis","ShortRunLowGrayLevelEmphasis"]] +
            [f"original_glszm_{n}" for n in ["GrayLevelNonUniformity","GrayLevelNonUniformityNormalized",
             "GrayLevelVariance","HighGrayLevelZoneEmphasis","LargeAreaEmphasis","LargeAreaHighGrayLevelEmphasis",
             "LargeAreaLowGrayLevelEmphasis","LowGrayLevelZoneEmphasis","SizeZoneNonUniformity",
             "SizeZoneNonUniformityNormalized","SmallAreaEmphasis","SmallAreaHighGrayLevelEmphasis",
             "SmallAreaLowGrayLevelEmphasis","ZoneEntropy","ZonePercentage","ZoneVariance"]] +
            [f"original_ngtdm_{n}" for n in ["Busyness","Coarseness","Complexity","Contrast","Strength"]] +
            [f"original_gldm_{n}" for n in ["DependenceEntropy","DependenceNonUniformity",
             "DependenceNonUniformityNormalized","DependenceVariance","GrayLevelNonUniformity",
             "GrayLevelVariance","HighGrayLevelEmphasis","LargeDependenceEmphasis",
             "LargeDependenceHighGrayLevelEmphasis","LargeDependenceLowGrayLevelEmphasis",
             "LowGrayLevelEmphasis","SmallDependenceEmphasis","SmallDependenceHighGrayLevelEmphasis",
             "SmallDependenceLowGrayLevelEmphasis"]] +
            [f"original_shape2D_{n}" for n in ["Elongation","MajorAxisLength","MaximumDiameter",
             "MeshSurface","MinorAxisLength","Perimeter","PerimeterSurfaceRatio",
             "PixelSurface","Sphericity"]]
        )
        return {name: float(rng.normal()) for name in feature_names}

    # ── Real PyRadiomics extraction ───────────────────────────────────────────
    img_uint = (img_arr_2d * 65535).astype(np.uint16)
    img_3d   = sitk.GetImageFromArray(img_uint[np.newaxis, :, :])
    mask_3d  = sitk.GetImageFromArray(mask_uint[np.newaxis, :, :])
    img_3d.SetSpacing((1.0, 1.0, 1.0))
    mask_3d.SetSpacing((1.0, 1.0, 1.0))

    try:
        result = _get_extractor().execute(img_3d, mask_3d, label=1)
    except Exception as e:
        log.debug("Radiomics failed for %s slice %d: %s", patient_id, slice_idx, e)
        return None

    out = {}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        fv = _to_float(v)
        if fv is not None and np.isfinite(fv):
            out[k] = fv
    return out or None


def build_slice_mask(seg_arr, slice_idx):
    """Extract and binarise the axial segmentation slice for a given index."""
    seg_slice = seg_arr[slice_idx]
    return (seg_slice > 0).astype(np.uint8)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def _extract_one_patient(job):
    """Worker: extract all slices for one patient. Returns list of row dicts."""
    pat_dir, kits_meta, mock = job
    patient_id = os.path.basename(pat_dir)
    from PIL import Image as PILImage

    paths = get_paths()
    kits19_dir = paths["kits19_data"]
    seg_path = os.path.join(kits19_dir, patient_id, "segmentation.nii.gz")
    if not os.path.exists(seg_path):
        return patient_id, None  # skip marker

    seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
    meta  = kits_meta.get(patient_id, {})
    label = isup_to_binary(meta.get("tumor_isup_grade") or meta.get("isup_grade"))

    rows = []
    patch_files = sorted(glob.glob(os.path.join(pat_dir, "slice_*.npy")))
    for patch_file in patch_files:
        slice_idx = int(os.path.basename(patch_file).replace("slice_", "").replace(".npy", ""))
        if slice_idx >= seg_arr.shape[0]:
            continue
        img_arr = np.load(patch_file)
        seg_slice = seg_arr[slice_idx]
        if not np.any(seg_slice > 0):
            continue
        coords = np.argwhere(seg_slice > 0)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        mask_crop = (seg_slice[r0:r1+1, c0:c1+1] > 0).astype(np.uint8)
        mask_resized = np.array(
            PILImage.fromarray(mask_crop).resize((128, 128), resample=PILImage.Resampling.NEAREST)
        )
        feats = extract_2d_radiomics(img_arr, mask_resized, patient_id, slice_idx, mock=mock)
        if feats is None:
            continue
        row = {"patient_id": patient_id, "slice_idx": slice_idx, "label": label}
        row.update(feats)
        rows.append(row)
    return patient_id, rows


def run_extraction(paths, kits_meta, mock=False, workers=None, resume=True):
    """Extract radiomics for all interim patients, with optional parallel workers.

    Checkpoints each patient to data/processed/radiomics_parts/<patient_id>.csv
    so interrupted runs resume without redoing finished patients.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    interim_dir = paths["interim"]
    parts_dir = os.path.join(paths["processed"], "radiomics_parts")
    os.makedirs(parts_dir, exist_ok=True)

    patient_dirs = sorted(glob.glob(os.path.join(interim_dir, "case_*")))
    if not patient_dirs:
        log.error("No cropped patches found in %s — run segment_and_crop.py first.", interim_dir)
        sys.exit(1)

    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    # Resume: skip patients already checkpointed
    todo = []
    loaded_parts = []
    for pat_dir in patient_dirs:
        pid = os.path.basename(pat_dir)
        part_path = os.path.join(parts_dir, f"{pid}.csv")
        done_path = part_path + ".done"
        if resume and os.path.exists(done_path):
            if os.path.exists(part_path) and os.path.getsize(part_path) > 0:
                try:
                    df_part = pd.read_csv(part_path)
                    if len(df_part) > 0:
                        loaded_parts.append(df_part)
                except Exception:
                    todo.append((pat_dir, kits_meta, mock))  # corrupt — redo
            # .done with empty/missing csv = patient fully processed, zero rows
            continue
        todo.append((pat_dir, kits_meta, mock))

    log.info(
        "Extracting radiomics: %d patients todo, %d already done (mock=%s, workers=%d)...",
        len(todo), len(patient_dirs) - len(todo), mock or not PYRADIOMICS_AVAILABLE, workers,
    )

    if not todo:
        log.info("Nothing to extract (all patients checkpointed).")
    elif workers <= 1 or mock:
        # Sequential (mock is fast; single-worker easier to debug)
        for job in tqdm(todo, desc="Patients"):
            pid, rows = _extract_one_patient(job)
            part_path = os.path.join(parts_dir, f"{pid}.csv")
            if rows is None:
                open(part_path + ".done", "w").close()
                continue
            pd.DataFrame(rows).to_csv(part_path, index=False)
            open(part_path + ".done", "w").close()
            loaded_parts.append(pd.DataFrame(rows))
    else:
        # Parallel by patient. Worker must be importable → run via this module.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_extract_one_patient, job): job[0] for job in todo}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Patients"):
                pat_dir = futures[fut]
                pid = os.path.basename(pat_dir)
                try:
                    _, rows = fut.result()
                except Exception as e:
                    log.error("Worker failed for %s: %s", pid, e)
                    continue
                part_path = os.path.join(parts_dir, f"{pid}.csv")
                if rows is None:
                    open(part_path + ".done", "w").close()
                    continue
                pd.DataFrame(rows).to_csv(part_path, index=False)
                open(part_path + ".done", "w").close()
                loaded_parts.append(pd.DataFrame(rows))

    if not loaded_parts:
        return pd.DataFrame()
    frames = [df for df in loaded_parts if df is not None and len(df) > 0]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["patient_id", "slice_idx"])


def select_top_k_features(df_train, k=TOP_K):
    """ANOVA F-value selection on labeled train rows only. Returns sorted feature names."""
    from sklearn.feature_selection import f_classif

    labeled = df_train.dropna(subset=["label"])
    if labeled.empty:
        log.warning("No labeled train rows — cannot run F-value selection. Returning first %d features.", k)
        feat_cols = [c for c in df_train.columns if c not in ("patient_id", "slice_idx", "label")]
        return feat_cols[:k]

    feat_cols = [c for c in labeled.columns if c not in ("patient_id", "slice_idx", "label")]
    X = labeled[feat_cols].fillna(0).values
    y = labeled["label"].values.astype(int)

    f_vals, _ = f_classif(X, y)
    ranked = sorted(zip(feat_cols, f_vals), key=lambda x: x[1] if np.isfinite(x[1]) else -1, reverse=True)
    top_k = [name for name, _ in ranked[:k]]
    log.info("Top-%d features selected by F-value: %s", k, top_k)
    return top_k


def normalize_features(df, feature_cols, scaler=None):
    """Z-score normalise. Fit scaler if None (train); otherwise transform only."""
    from sklearn.preprocessing import StandardScaler
    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols].fillna(0))
    else:
        df[feature_cols] = scaler.transform(df[feature_cols].fillna(0))
    return df, scaler


def main():
    parser = argparse.ArgumentParser(description="Step 3: PyRadiomics extraction + top-16 F-value selection")
    parser.add_argument("--splits", default=None, help="Path to splits.json (from Step 5). If omitted, runs unsplit.")
    parser.add_argument("--mock",   action="store_true",
                        help="Use synthetic features instead of pyradiomics (auto-set if pyradiomics unavailable).")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel patient workers (default: cpu_count-1, max 8).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing radiomics_parts checkpoints and re-extract everything.")
    args = parser.parse_args()

    mock = args.mock or not PYRADIOMICS_AVAILABLE
    paths = get_paths()
    os.makedirs(paths["processed"], exist_ok=True)

    # Load kits.json labels
    kits_meta = load_kits_json(paths["kits_json"])
    log.info("Loaded metadata for %d patients.", len(kits_meta))

    # Run radiomics extraction
    df = run_extraction(paths, kits_meta, mock=mock,
                        workers=args.workers, resume=not args.no_resume)
    if df.empty:
        log.error("No features extracted. Aborting.")
        sys.exit(1)

    out_all = os.path.join(paths["processed"], "radiomics_all.csv")
    df.to_csv(out_all, index=False)
    log.info("Saved all features → %s (%d rows, %d cols)", out_all, len(df), len(df.columns))

    # ── Feature selection (train-only, if splits provided) ───────────────────
    if args.splits and os.path.exists(args.splits):
        with open(args.splits) as f:
            splits = json.load(f)
        train_ids = set(splits.get("train", []))
        df_train  = df[df["patient_id"].isin(train_ids)]
    else:
        log.warning("No splits.json provided (Step 5 not done yet). Running F-value selection on ALL data as pilot.")
        df_train = df

    feature_cols = [c for c in df.columns if c not in ("patient_id", "slice_idx", "label")]
    top16 = select_top_k_features(df_train, k=TOP_K)

    # Freeze feature list
    top16_path = os.path.join(paths["processed"], "top16_features.json")
    with open(top16_path, "w") as f:
        json.dump(top16, f, indent=2)
    log.info("Frozen top-16 feature list → %s", top16_path)

    # Normalise: fit on train, apply to all
    df_top = df[["patient_id", "slice_idx", "label"] + top16].copy()
    if args.splits and os.path.exists(args.splits):
        df_train_top = df_top[df_top["patient_id"].isin(train_ids)].copy()
        df_other     = df_top[~df_top["patient_id"].isin(train_ids)].copy()
        df_train_top, scaler = normalize_features(df_train_top, top16)
        if len(df_other) > 0:   # scaler.transform errors on 0-sample frames (empty val/test)
            df_other, _ = normalize_features(df_other, top16, scaler=scaler)
        df_top = pd.concat([df_train_top, df_other]).sort_values(["patient_id", "slice_idx"])
    else:
        df_top, scaler = normalize_features(df_top, top16)

    out_top16 = os.path.join(paths["processed"], "radiomics_top16.csv")
    df_top.to_csv(out_top16, index=False)
    log.info("Saved top-16 features → %s", out_top16)

    scaler_path = os.path.join(paths["processed"], "radiomics_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    log.info("Saved StandardScaler → %s", scaler_path)

    log.info("Step 3 complete.")


if __name__ == "__main__":
    main()
