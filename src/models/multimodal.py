"""
Task 10: Multimodal Fusion Model
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn
from src.models.image_encoder import ImageEncoder3D
from src.models.clinical_encoder import ClinicalEncoder


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes=4, image_feature_dim=256,
                 clinical_feature_dim=128, dropout=0.3):
        super().__init__()
        self.image_encoder = ImageEncoder3D(in_channels=1, feature_dim=image_feature_dim)
        self.clinical_encoder = ClinicalEncoder(input_dim=32, feature_dim=clinical_feature_dim, dropout=dropout)

        fused_dim = image_feature_dim + clinical_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, clinical):
        img_feat = self.image_encoder(image)
        clin_feat = self.clinical_encoder(clinical)
        fused = torch.cat([img_feat, clin_feat], dim=1)
        return self.classifier(fused)

    def predict(self, image, clinical):
        self.eval()
        with torch.no_grad():
            logits = self.forward(image, clinical)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        return preds, probs


if __name__ == "__main__":
    model = MultimodalClassifier(num_classes=4, dropout=0.3)
    img = torch.randn(2, 1, 128, 128, 128)
    clin = torch.randn(2, 32)
    out = model(img, clin)
    print(f"Image input: {img.shape}")
    print(f"Clinical input: {clin.shape}")
    print(f"Output logits: {out.shape}")
    print(f"Output probs: {torch.softmax(out, dim=1).shape}")

    preds, probs = model.predict(img, clin)
    print(f"Predictions: {preds}")
    print(f"Probabilities: {probs}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
