"""
Task 15: Baseline Comparison Models
1. Image-only model (3D CNN without clinical data)
2. Clinical-only model (MLP without images)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from src.models.image_encoder import ImageEncoder3D
from src.models.clinical_encoder import ClinicalEncoder


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=4, feature_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = ImageEncoder3D(in_channels=1, feature_dim=feature_dim)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, clinical=None):
        feat = self.encoder(image)
        return self.classifier(feat)


class ClinicalOnlyModel(nn.Module):
    def __init__(self, num_classes=4, input_dim=32, feature_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = ClinicalEncoder(input_dim=input_dim, feature_dim=feature_dim, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, image=None, clinical=None):
        feat = self.encoder(clinical)
        return self.classifier(feat)


def test_model(model, name, num_classes=4):
    if name == "Image-Only":
        img = torch.randn(2, 1, 128, 128, 128)
        clin = None
    else:
        img = None
        clin = torch.randn(2, 32)

    model.eval()
    with torch.no_grad():
        out = model(img, clin)

    probs = torch.softmax(out, dim=1)
    preds = torch.argmax(probs, dim=1)
    params = sum(p.numel() for p in model.parameters())

    print(f"{name}:")
    print(f"  Output shape: {out.shape}")
    print(f"  Predictions: {preds.tolist()}")
    print(f"  Probabilities: {probs.numpy().round(3)}")
    print(f"  Parameters: {params:,}")
    return params


if __name__ == "__main__":
    print("=== Baseline Models ===\n")

    img_model = ImageOnlyModel(num_classes=4)
    img_params = test_model(img_model, "Image-Only")

    print()

    clin_model = ClinicalOnlyModel(num_classes=4)
    clin_params = test_model(clin_model, "Clinical-Only")

    print(f"\nSummary:")
    print(f"  Image-Only: {img_params:,} params")
    print(f"  Clinical-Only: {clin_params:,} params")
    print(f"  Multimodal (from Task 10): 3,741,156 params")
    print("\nBaseline models test passed!")
