"""
Baseline fusion: majority voting across the 6 sector predictions (plan §4 Step 7.6)
====================================================================================
Paper baseline: each sector head (incl. class token) casts a hard vote at
threshold 0.5 on its sigmoid probability; the final prediction is the majority.

Tie handling (paper under-specifies; documented as our choice): with 6 voters a
3–3 tie is possible — broken by the class-token vote (first token in TOKEN_ORDER).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn as nn

from src.models.vvit import VViT, TOKEN_ORDER

CLASS_IDX = TOKEN_ORDER.index("class")  # 0


def majority_vote(logits: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    logits : (B, 6) raw sector logits in TOKEN_ORDER
             [class, demographic, comorbidity, habit, radiomic, image]

    Returns
    -------
    preds : (B,) long — 0 (low) or 1 (high), majority vote with class-token tie-break
    """
    if logits.dim() != 2 or logits.shape[1] != len(TOKEN_ORDER):
        raise ValueError(f"expected (B, {len(TOKEN_ORDER)}) logits, got {tuple(logits.shape)}")
    votes = (torch.sigmoid(logits) >= 0.5).long()          # (B, 6)
    vote_sum = votes.sum(dim=1)                             # (B,)
    n = votes.shape[1]
    half = n // 2                                           # 3 for n=6
    preds = (vote_sum > half).long()                        # strict majority >= 4
    # 3–3 ties (vote_sum == half): fall back to class token
    tie = vote_sum == half
    if tie.any():
        preds[tie] = votes[tie, CLASS_IDX]
    return preds


class MajorityVoteFusion(nn.Module):
    """VViT + majority-voting head — paper baseline for Table 2 / total-model metrics."""

    def __init__(self, **vvit_kwargs):
        super().__init__()
        self.vvit = VViT(**vvit_kwargs)

    def forward(self, batch: dict) -> torch.Tensor:
        """Returns (B, 6) sector logits (same as VViT)."""
        return self.vvit(batch)

    @torch.no_grad()
    def predict(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        preds  : (B,) majority-vote binary predictions
        probs  : (B, 6) per-sector sigmoid probabilities (for metrics/aggregation)
        """
        logits = self.vvit(batch)
        probs = torch.sigmoid(logits)
        return majority_vote(logits), probs


if __name__ == "__main__":
    torch.manual_seed(0)
    B = 8

    # All sectors agree low
    low = torch.full((B, 6), -4.0)
    assert torch.equal(majority_vote(low), torch.zeros(B, dtype=torch.long))

    # All sectors agree high
    high = torch.full((B, 6), 4.0)
    assert torch.equal(majority_vote(high), torch.ones(B, dtype=torch.long))

    # 4 high / 2 low -> high (strict majority)
    m = torch.full((B, 6), -4.0)
    m[:, :4] = 4.0
    assert torch.equal(majority_vote(m), torch.ones(B, dtype=torch.long))

    # 3–3 tie broken by class token (token 0) high
    t = torch.full((B, 6), -4.0)
    t[:, :3] = 4.0  # class + demo + comorb high; rest low
    assert torch.equal(majority_vote(t), torch.ones(B, dtype=torch.long))
    # class low, others make 3 high elsewhere -> class breaks tie low
    t2 = torch.full((B, 6), -4.0)
    t2[:, 0] = -4.0          # class low
    t2[:, 1:4] = 4.0         # demo, comorb, habit high -> 3 high, 3 low
    assert torch.equal(majority_vote(t2), torch.zeros(B, dtype=torch.long))

    # End-to-end with VViT on dummy batch
    fusion = MajorityVoteFusion()
    fusion.eval()
    batch = {
        "demographic": torch.randn(B, 3),
        "comorbidity": torch.randn(B, 3),
        "habit":       torch.randn(B, 2),
        "radiomic":    torch.randn(B, 16),
        "image":       torch.rand(B, 1, 128, 128),
    }
    preds, probs = fusion.predict(batch)
    assert preds.shape == (B,) and probs.shape == (B, 6)
    assert set(preds.unique().tolist()) <= {0, 1}

    print(f"majority_vote edge cases OK (n=6, tie-break=class token '{TOKEN_ORDER[CLASS_IDX]}')")
    print(f"MajorityVoteFusion.predict: preds {tuple(preds.shape)}, probs {tuple(probs.shape)}")
    print("fusion_baseline_vote.py self-test PASSED.")
