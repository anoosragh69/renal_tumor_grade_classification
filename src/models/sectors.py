"""
Sector tokenizers and per-sector classification heads (plan §4 Step 7.1, 7.5).
==============================================================================
Each raw sector vector is linearly projected into the shared embedding dim;
each of the 6 encoder output tokens gets its own binary classification head.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn


class SectorTokenizer(nn.Module):
    """Linear projection of one raw sector vector -> shared embedding dim."""

    def __init__(self, sector_dim: int, embed_dim: int):
        super().__init__()
        self.sector_dim = sector_dim
        self.embed_dim = embed_dim
        self.proj = nn.Linear(sector_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, sector_dim) -> (B, embed_dim)
        return self.proj(x)


class PerSectorHead(nn.Module):
    """Binary logit head for one token of the encoder output sequence."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        # token: (B, embed_dim) -> (B,) logits
        return self.head(token).squeeze(-1)


if __name__ == "__main__":
    tok = SectorTokenizer(sector_dim=16, embed_dim=128)
    head = PerSectorHead(embed_dim=128)
    x = torch.randn(4, 16)
    z = tok(x)
    logit = head(z)
    assert z.shape == (4, 128), z.shape
    assert logit.shape == (4,), logit.shape
    print(f"SectorTokenizer: {tuple(x.shape)} -> {tuple(z.shape)}")
    print(f"PerSectorHead:   {tuple(z.shape)} -> {tuple(logit.shape)}")
    print("sectors.py self-test PASSED.")
