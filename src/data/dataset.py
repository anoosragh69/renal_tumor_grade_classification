"""
Step 6: Sector-dict Dataset & DataLoader (plan §4 Step 6)
===========================================================
Rewrite of the old Task 11 (3D tuple) dataset for the vViT multi-sector pipeline.

Sector-dict sample (model-facing keys match plan §4 Step 6):

    {
      "demographic": FloatTensor(3),        # age, sex, bmi   (z-scored by Step 4)
      "comorbidity": FloatTensor(3),        # pvd, dm, ckd    (synthetic, label-independent)
      "habit":       FloatTensor(2),        # smoking, alcohol(synthetic, label-independent)
      "radiomic":    FloatTensor(16),       # frozen top-16, train-fit z-scored (Step 3)
      "image":       FloatTensor(1,128,128),# HU-windowed, [0,1]; model flattens to 16384
      "label":       FloatTensor(()),       # 0=low (ISUP 1-2), 1=high (ISUP 3-4); float for BCE
      # metadata (NOT model inputs — for patient-level aggregation & permutation importance):
      "patient_id":  str,
      "slice_idx":   LongTensor(()),
    }

Sample granularity = one 2D tumour slice (patient_id, slice_idx); clinical sectors repeat
across a patient's slices. Slices are included only when they exist in BOTH
data/interim (Step 2 crop) and radiomics_top16.csv (Step 3 extraction succeeded).

Augmentations — train split, IMAGE SECTOR ONLY (plan §4 Step 6):
    horizontal flip, vertical flip, perspective, invert, posterize, solarize, equalize
    (torchvision.transforms; each applied with p=0.5 — the paper under-specifies
    probabilities, document as our choice). Applied in uint8 PIL space, which
    posterize/equalize require; train images therefore pass through 8-bit quantisation.

Usage:
    python src/data/dataset.py            # self-test on current data/ artifacts
    loaders = get_dataloaders(batch_size=8)  # {'train': ..., 'val': ..., 'test': ...}
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as T

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Sector schema (handoff contract for Teammate B) ───────────────────────────
SECTOR_DIMS = {
    "demographic": 3,
    "comorbidity": 3,
    "habit":       2,
    "radiomic":    16,
    "image":       128 * 128,   # flattened dim the model sees (plan §4 Step 7)
}
IMAGE_SHAPE = (1, 128, 128)
SECTOR_KEYS = ("demographic", "comorbidity", "habit", "radiomic", "image")
META_KEYS = ("patient_id", "slice_idx")
LABEL_KEY = "label"

DEMOGRAPHIC_COLS = ["age", "sex", "bmi"]
COMORBIDITY_COLS = ["pvd", "dm", "ckd"]
HABIT_COLS = ["smoking", "alcohol"]


def get_paths(root=REPO_ROOT):
    processed = os.path.join(root, "data", "processed")
    return {
        "splits":    os.path.join(processed, "splits.json"),
        "clinical":  os.path.join(processed, "clinical.csv"),
        "radiomics": os.path.join(processed, "radiomics_top16.csv"),
        "top16":     os.path.join(processed, "top16_features.json"),
        "interim":   os.path.join(root, "data", "interim"),
    }


def build_paper_augmentation(p=0.5):
    """Plan §4 Step 6 augmentation list, image sector only, uint8 PIL space."""
    return T.Compose([
        T.RandomHorizontalFlip(p=p),
        T.RandomVerticalFlip(p=p),
        T.RandomPerspective(distortion_scale=0.5, p=p, fill=0),
        T.RandomInvert(p=p),
        T.RandomPosterize(bits=4, p=p),
        T.RandomSolarize(threshold=128, p=p),
        T.RandomEqualize(p=p),
    ])


class RenalTumorSectorDataset(Dataset):
    """Sector-dict dataset over one split (train / val / test)."""

    def __init__(self, split="train", augment=None, paths=None, root=REPO_ROOT):
        if paths is None:
            paths = get_paths(root)
        self.split = split
        # augment defaults to train-only
        self.augment = (split == "train") if augment is None else augment
        self.paths = paths
        self.transform = build_paper_augmentation() if self.augment else None

        # ── Split membership (versioned artifact, Step 5) ─────────────────────
        with open(paths["splits"]) as f:
            splits = json.load(f)
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.patient_ids = set(splits.get(split, []))

        # ── Clinical sectors + labels (Step 4) ────────────────────────────────
        clinical = pd.read_csv(paths["clinical"])
        clinical = clinical[clinical["patient_id"].isin(self.patient_ids)]
        missing = self.patient_ids - set(clinical["patient_id"])
        if missing:
            raise ValueError(f"Split patients missing from clinical.csv: {sorted(missing)}")
        clinical = clinical.set_index("patient_id")
        self.clinical = {
            pid: (
                row[DEMOGRAPHIC_COLS].to_numpy(dtype=np.float32),
                row[COMORBIDITY_COLS].to_numpy(dtype=np.float32),
                row[HABIT_COLS].to_numpy(dtype=np.float32),
                float(row["label"]),
            )
            for pid, row in clinical.iterrows()
        }

        # ── Radiomic vectors (frozen top-16 order, Step 3) ────────────────────
        with open(paths["top16"]) as f:
            self.top16 = json.load(f)
        radiomics = pd.read_csv(paths["radiomics"])
        radiomics = radiomics[radiomics["patient_id"].isin(self.patient_ids)]
        self.radiomics = {
            (r["patient_id"], int(r["slice_idx"])): r[self.top16].to_numpy(dtype=np.float32)
            for _, r in radiomics.iterrows()
        }

        # ── Slice index: intersection of interim crops (Step 2) and radiomics ─
        self.index = []   # list of (patient_id, slice_idx)
        skipped_missing_npy = 0
        for pid, slice_idx in self.radiomics:
            npy = os.path.join(paths["interim"], pid, f"slice_{slice_idx:04d}.npy")
            if os.path.exists(npy):
                self.index.append((pid, slice_idx))
            else:
                skipped_missing_npy += 1
        self.index.sort()
        if skipped_missing_npy:
            print(f"WARNING [{split}]: {skipped_missing_npy} radiomics rows have no interim .npy — skipped.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pid, slice_idx = self.index[idx]
        demo, comorb, habit, label = self.clinical[pid]

        # Image sector — float [0,1] crop from Step 2 (HU window −400..400)
        img = np.load(os.path.join(self.paths["interim"], pid, f"slice_{slice_idx:04d}.npy"))
        img = img.astype(np.float32)

        if self.transform is not None:
            # posterize/equalize need uint8; operate in PIL, return float [0,1]
            pil = Image.fromarray((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
            pil = self.transform(pil)
            img = np.asarray(pil, dtype=np.float32) / 255.0

        return {
            "demographic": torch.from_numpy(demo),
            "comorbidity": torch.from_numpy(comorb),
            "habit":       torch.from_numpy(habit),
            "radiomic":    torch.from_numpy(self.radiomics[(pid, slice_idx)]),
            "image":       torch.from_numpy(img).unsqueeze(0),   # (1,128,128)
            LABEL_KEY:     torch.tensor(label, dtype=torch.float32),
            # metadata for patient-level aggregation (Step 9) / permutation importance (Step 10)
            "patient_id":  pid,
            "slice_idx":   torch.tensor(slice_idx, dtype=torch.long),
        }


def get_dataloaders(batch_size=8, num_workers=0, augment=True, seed=42, root=REPO_ROOT):
    """
    Returns {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}.
    Train is shuffled with a seeded generator; val/test are deterministic.
    Empty splits yield empty loaders (0 batches) rather than erroring.
    """
    loaders = {}
    for split in ("train", "val", "test"):
        ds = RenalTumorSectorDataset(
            split=split,
            augment=(split == "train" and augment),
            root=root,
        )
        shuffle = split == "train" and len(ds) > 0
        gen = torch.Generator().manual_seed(seed) if shuffle else None
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=gen,
            drop_last=False,
        )
    return loaders


def _self_test():
    """Verify schema, augmentation behaviour, determinism, label consistency."""
    paths = get_paths()

    loaders = get_dataloaders(batch_size=4, augment=True)
    print("Split sizes (patients / slices):")
    for split, dl in loaders.items():
        ds = dl.dataset
        print(f"  {split:5s}: {len(ds.patient_ids)} patients, {len(ds)} slices, {len(dl)} batches")

    # ── Batch schema check on train ───────────────────────────────────────────
    train = loaders["train"]
    if len(train) == 0:
        raise AssertionError("train loader is empty — run Steps 2-5 first")
    batch = next(iter(train))
    B = batch["image"].shape[0]
    expected = {
        "demographic": (B, 3), "comorbidity": (B, 3), "habit": (B, 2),
        "radiomic": (B, 16), "image": (B, 1, 128, 128), "label": (B,),
        "slice_idx": (B,),
    }
    for k, shape in expected.items():
        assert tuple(batch[k].shape) == shape, f"{k}: {tuple(batch[k].shape)} != {shape}"
        assert batch[k].dtype == torch.float32 or k == "slice_idx", f"{k} dtype {batch[k].dtype}"
    assert len(batch["patient_id"]) == B
    print(f"Batch schema OK: { {k: tuple(v.shape) for k, v in batch.items() if k != 'patient_id'} }")

    # ── Augmentation: train images should differ from source crops; range OK ──
    ds_tr = train.dataset
    diffs = 0
    n_check = min(16, len(ds_tr))
    for i in range(n_check):
        item = ds_tr[i]
        assert 0.0 <= item["image"].min() and item["image"].max() <= 1.0, "image out of [0,1]"
        pid, sidx = ds_tr.index[i]
        src = np.load(os.path.join(paths["interim"], pid, f"slice_{sidx:04d}.npy"))
        if not np.allclose(item["image"].squeeze(0).numpy(), src, atol=1 / 255 + 1e-6):
            diffs += 1
    print(f"Augmentation active on train: {diffs}/{n_check} samples differ from source crop "
          f"(7 ops each p=0.5 -> expected ~all)")

    # ── No augmentation on val/test: exact reconstruction ─────────────────────
    for split in ("val", "test"):
        ds = RenalTumorSectorDataset(split=split, augment=None)   # non-train default: off
        for i in range(len(ds)):
            item = ds[i]
            pid, sidx = ds.index[i]
            src = np.load(os.path.join(paths["interim"], pid, f"slice_{sidx:04d}.npy"))
            assert np.allclose(item["image"].squeeze(0).numpy(), src, atol=1e-6), \
                f"{split} item {i} altered without augmentation"
    print("Val/test determinism OK (images bit-exact vs source crops)")

    # ── Label consistency with clinical.csv ───────────────────────────────────
    clinical = pd.read_csv(paths["clinical"]).set_index("patient_id")
    for i in range(len(ds_tr)):
        item = ds_tr[i]
        assert item["label"].item() == float(clinical.loc[item["patient_id"], "label"]), \
            "label mismatch vs clinical.csv"
    print("Labels consistent with clinical.csv")

    # ── Split disjointness (leakage guard) ────────────────────────────────────
    sets = {s: set(loaders[s].dataset.patient_ids) for s in ("train", "val", "test")}
    assert not (sets["train"] & sets["val"]),   "train/val patient overlap!"
    assert not (sets["train"] & sets["test"]),  "train/test patient overlap!"
    assert not (sets["val"] & sets["test"]),    "val/test patient overlap!"
    print("Patient-level split disjointness OK")

    print("\nStep 6 self-test PASSED.")


if __name__ == "__main__":
    _self_test()
