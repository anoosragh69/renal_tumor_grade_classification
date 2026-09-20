"""
Task 9: Clinical Feature Encoder (MLP)
"""
import torch
import torch.nn as nn


class ClinicalEncoder(nn.Module):
    def __init__(self, input_dim=32, feature_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    model = ClinicalEncoder(input_dim=32, feature_dim=128)
    x = torch.randn(4, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
