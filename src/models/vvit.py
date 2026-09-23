"""
vViT multi-sector transformer skeleton (plan §4 Step 7 / Fig. 3)
================================================================
Architecture (paper hyperparams):
  - sector tokenizers: linear projection per sector -> shared embed dim 128
  - class token: learnable embedding prepended to the sector sequence
  - sequence: [cls, demographic, comorbidity, habit, radiomic, image] -> (B, 6, 128)
  - pre-norm transformer encoder: depth 8, heads 8, MLP dim 32
  - per-sector heads: each of the 6 output tokens -> own linear binary head
  - output: (B, 6) sector logits (BCE applied jointly in Step 7c)

head_dim note: embed 128 / 8 heads = head_dim 16 under standard MHA.
The plan's "head dim 64" is incompatible with embed/heads; we prioritize
embed/heads/depth and document head_dim 16 as a report deviation.

Usage:
    python src/models/vvit.py            # dummy-tensor self-test
    model = VViT()
    logits = model(sector_batch)         # sector_batch from Step 6 DataLoader
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from src.data.dataset import SECTOR_DIMS, SECTOR_KEYS, IMAGE_SHAPE
from src.models.sectors import SectorTokenizer, PerSectorHead

# Shared sequence order: class token first, then the 5 data sectors (plan §4 Step 7.3)
TOKEN_ORDER = ("class",) + SECTOR_KEYS  # ("class", "demographic", ..., "image")
NUM_TOKENS = len(TOKEN_ORDER)           # 6


class VViT(nn.Module):
    """Variable Vision Transformer over 5 clinical/radiomic/image sectors + class token."""

    def __init__(
        self,
        sector_dims: dict | None = None,
        embed_dim: int = 128,
        depth: int = 8,
        num_heads: int = 8,
        mlp_dim: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if sector_dims is None:
            sector_dims = SECTOR_DIMS
        self.sector_dims = dict(sector_dims)
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.token_order = TOKEN_ORDER

        # Learnable class token (plan §4 Step 7.2)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.class_token, std=0.02)

        # Per-sector linear tokenizers (plan §4 Step 7.1)
        self.tokenizers = nn.ModuleDict({
            key: SectorTokenizer(self.sector_dims[key], embed_dim)
            for key in SECTOR_KEYS
        })

        # Pre-norm transformer encoder (plan §4 Step 7.4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,          # pre-norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Per-sector binary heads (plan §4 Step 7.5) — one per token incl. class
        self.heads = nn.ModuleDict({
            tok: PerSectorHead(embed_dim) for tok in TOKEN_ORDER
        })

    def _tokenize(self, batch: dict) -> torch.Tensor:
        """Build (B, 6, 128) sequence: [class, demographic, comorbidity, habit, radiomic, image]."""
        tokens = []
        B = None
        for key in SECTOR_KEYS:
            x = batch[key]
            if x.dim() > 2:
                # image arrives as (B, 1, 128, 128) -> flatten to (B, 16384)
                x = x.flatten(1)
            z = self.tokenizers[key](x)
            tokens.append(z)
            B = z.shape[0]
        assert B is not None, "batch has no sector keys"
        cls = self.class_token.expand(B, -1, -1)   # (1,1,E) -> (B,1,E)
        return torch.cat([cls, torch.stack(tokens, dim=1)], dim=1)  # (B, 6, E)

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict with SECTOR_KEYS tensors (Step 6 sector-dict sample);
                image may be (B,1,128,128) or already flattened (B,16384).

        Returns
        -------
        logits : (B, 6) — one binary logit per token in TOKEN_ORDER
                 order = [class, demographic, comorbidity, habit, radiomic, image]
        """
        x = self._tokenize(batch)                 # (B, 6, 128)
        x = self.encoder(x)                       # (B, 6, 128)
        x = self.encoder_norm(x)
        logits = torch.stack(
            [self.heads[tok](x[:, i]) for i, tok in enumerate(self.token_order)],
            dim=1,
        )                                          # (B, 6)
        return logits

    @torch.no_grad()
    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Sigmoid over sector logits -> (B, 6) probabilities in [0,1]."""
        return torch.sigmoid(self.forward(batch))


def _make_dummy_batch(B: int = 4, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "demographic": torch.randn(B, 3, generator=g),
        "comorbidity": torch.randn(B, 3, generator=g),
        "habit":       torch.randn(B, 2, generator=g),
        "radiomic":    torch.randn(B, 16, generator=g),
        "image":       torch.rand(B, *IMAGE_SHAPE, generator=g),
        "label":       torch.randint(0, 2, (B,), generator=g).float(),
    }


def _self_test():
    """Verify shapes, dtypes, grad flow, and SECTOR_DIMS contract on dummy tensors."""
    print("=== vViT self-test (dummy tensors) ===")
    model = VViT()
    model.train()
    batch = _make_dummy_batch(B=4)

    # ── Contract: sector dims match Step 6 ───────────────────────────────────
    for key, dim in SECTOR_DIMS.items():
        assert key in batch, f"missing sector {key}"
        if key == "image":
            assert batch[key][0].numel() == dim, \
                f"image flat size {batch[key][0].numel()} != SECTOR_DIMS {dim}"
        else:
            assert batch[key].shape[1] == dim, \
                f"{key} dim {batch[key].shape[1]} != SECTOR_DIMS {dim}"
    print(f"SECTOR_DIMS contract OK: {SECTOR_DIMS}")

    # ── Forward shapes ───────────────────────────────────────────────────────
    logits = model(batch)
    assert logits.shape == (4, NUM_TOKENS), f"logits {tuple(logits.shape)} != (4, {NUM_TOKENS})"
    assert logits.dtype == torch.float32
    print(f"Forward: sector batch -> logits {tuple(logits.shape)} "
          f"(order={model.token_order})")

    # ── Flattened image also accepted ────────────────────────────────────────
    flat_batch = dict(batch)
    flat_batch["image"] = batch["image"].flatten(1)
    logits_flat = model(flat_batch)
    assert logits_flat.shape == (4, NUM_TOKENS)
    assert torch.allclose(logits, logits_flat, atol=1e-6), \
        "flattened vs (1,128,128) image input should match"
    print("Image reshape path OK (flat == unflat)")

    # ── Grad flow through shared encoder + heads ─────────────────────────────
    model.zero_grad(set_to_none=True)
    loss = logits.sum()
    loss.backward()
    grad_ok = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"no grad for {name}"
            grad_ok.append(name)
    # class token must receive gradient
    assert model.class_token.grad is not None, "class_token has no grad"
    assert model.class_token.grad.abs().sum() > 0, "class_token grad is zero"
    print(f"Grad flow OK ({len(grad_ok)} params; class_token grad "
          f"norm={model.class_token.grad.norm():.4f})")

    # ── Determinism in eval mode ─────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        a = model(batch)
        b = model(batch)
    assert torch.equal(a, b), "eval forward not deterministic"
    print("Eval determinism OK")

    # ── Param count ──────────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total ({n_train:,} trainable)")
    print(f"Encoder: depth={model.depth} heads={model.num_heads} "
          f"embed={model.embed_dim} mlp={model.mlp_dim} "
          f"(head_dim={model.embed_dim // model.num_heads})")
    print("\nvViT self-test PASSED.")


if __name__ == "__main__":
    _self_test()
