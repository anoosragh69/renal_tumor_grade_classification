"""
Task 8: 3D CNN Image Feature Extractor
"""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ImageEncoder3D(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock3D(in_channels, 32),
            nn.MaxPool3d(2),
            ConvBlock3D(32, 64),
            nn.MaxPool3d(2),
            ConvBlock3D(64, 128),
            nn.MaxPool3d(2),
            ConvBlock3D(128, 256),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)


if __name__ == "__main__":
    model = ImageEncoder3D(in_channels=1, feature_dim=256)
    x = torch.randn(2, 1, 128, 128, 128)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
