"""
Task 11: PyTorch Dataset & DataLoader for KiTS19 Multimodal Data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.data.preprocess import preprocess_ct_volume
from src.data.clinical import preprocess as preprocess_clinical


DATA_DIR = "/mnt/SharedDrive/Projects/renal_tumor_grade_classification/kits19/data"


class KiTS19Dataset(Dataset):
    def __init__(self, case_ids, labels, clinical_features,
                 data_dir=DATA_DIR, target_size=(128, 128, 128),
                 augment=False):
        self.case_ids = case_ids
        self.labels = labels
        self.clinical = clinical_features
        self.data_dir = data_dir
        self.target_size = target_size
        self.augment = augment

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, idx):
        case_id = self.case_ids[idx]
        label = self.labels[idx]
        clinical = self.clinical[idx]

        imaging_path = os.path.join(self.data_dir, case_id, "imaging.nii.gz")
        segmentation_path = os.path.join(self.data_dir, case_id, "segmentation.nii.gz")

        volume = preprocess_ct_volume(
            imaging_path, segmentation_path,
            target_size=self.target_size,
            use_segmentation=True
        )

        volume = volume.astype(np.float32)
        volume = np.expand_dims(volume, axis=0)

        if self.augment:
            volume = self._augment(volume)

        return (
            torch.tensor(volume, dtype=torch.float32),
            torch.tensor(clinical, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )

    def _augment(self, vol):
        # vol shape: (1, D, H, W) - 4D
        if np.random.random() > 0.5:
            vol = np.flip(vol, axis=1).copy()
        if np.random.random() > 0.5:
            vol = np.flip(vol, axis=2).copy()
        if np.random.random() > 0.5:
            vol = np.flip(vol, axis=3).copy()
        if np.random.random() > 0.3:
            vol = vol + np.random.normal(0, 0.05, vol.shape).astype(np.float32)
        return vol


def get_dataloaders(batch_size=2, num_workers=0):
    clinical = preprocess_clinical()
    splits = clinical["splits"]
    loaders = {}

    for split_name, (x, y, ids) in splits.items():
        dataset = KiTS19Dataset(
            case_ids=ids,
            labels=y,
            clinical_features=x,
            augment=(split_name == "train"),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
        )

    return loaders, clinical["feature_names"], clinical["num_classes"]


if __name__ == "__main__":
    print("Testing Dataset & DataLoader...")
    loaders, feat_names, num_classes = get_dataloaders(batch_size=2)

    for split_name, loader in loaders.items():
        batch = next(iter(loader))
        img, clin, lbl = batch
        print(f"\n{split_name.upper()}:")
        print(f"  Image batch: {img.shape}")
        print(f"  Clinical batch: {clin.shape}")
        print(f"  Labels: {lbl}")
        print(f"  Batches: {len(loader)}")

    print(f"\nFeatures: {len(feat_names)}")
    print(f"Classes: {num_classes}")
    print("Dataset & DataLoader test passed!")
