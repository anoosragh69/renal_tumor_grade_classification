"""
Training helpers: multi-sector BCE + paper Adam hyperparams (plan §4 Step 7.7–7.8)
==================================================================================
Loss (documented choice — paper under-specifies aggregation):
  binary_cross_entropy_with_logits on every sector head vs the shared binary label,
  averaged over batch and the 6 tokens (equivalent to mean of per-sector BCE means).
  Gradients flow jointly through the shared encoder.

Optimizer (exact paper match):
  Adam, β1=0.9, β2=0.999, ε=1e-8, weight_decay=0, AMSGrad=False
  (lr not specified by the paper — default 1e-3, configurable; report as our choice).

Usage:
    python src/training/train.py           # self-test: loss shapes + optimizer config
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F


def multi_sector_bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    logits : (B, 6) raw sector logits from VViT
    labels : (B,) float/binary {0,1}

    Returns
    -------
    scalar loss — mean BCE over batch × sectors
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be (B, S), got {tuple(logits.shape)}")
    if labels.shape[0] != logits.shape[0]:
        raise ValueError(f"batch mismatch: logits {tuple(logits.shape)} vs labels {tuple(labels.shape)}")
    target = labels.float().view(-1, 1).expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


def multi_sector_bce_per_head(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sector mean BCE, shape (S,) — for logging individual head losses."""
    target = labels.float().view(-1, 1).expand_as(logits)
    return F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=0)


def build_paper_adam(parameters, lr: float = 1e-3) -> torch.optim.Adam:
    """Adam with the paper's exact hyperparameters (weight_decay=0, amsgrad=False)."""
    return torch.optim.Adam(
        parameters,
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
    )


def _self_test():
    from src.models.vvit import VViT, _make_dummy_batch
    from src.models.fusion_baseline_vote import majority_vote

    print("=== training helpers self-test ===")
    torch.manual_seed(0)
    model = VViT()
    batch = _make_dummy_batch(B=4)
    labels = batch["label"]

    logits = model(batch)
    assert logits.shape == (4, 6)

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss = multi_sector_bce_loss(logits, labels)
    assert loss.dim() == 0 and torch.isfinite(loss), loss
    per_head = multi_sector_bce_per_head(logits, labels)
    assert per_head.shape == (6,)
    # mean of per-head == joint mean (same reduction)
    assert torch.allclose(loss, per_head.mean(), atol=1e-6)
    print(f"BCE joint={loss.item():.4f}; per-head={per_head.tolist()}")

    # ── Backward through shared encoder ──────────────────────────────────────
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert model.class_token.grad is not None and model.class_token.grad.abs().sum() > 0
    print("Loss backprop reaches class_token + shared encoder")

    # ── Adam paper hyperparams ───────────────────────────────────────────────
    opt = build_paper_adam(model.parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.Adam)
    b1, b2 = opt.defaults["betas"]
    assert abs(b1 - 0.9) < 1e-12 and abs(b2 - 0.999) < 1e-12
    assert abs(opt.defaults["eps"] - 1e-8) < 1e-18
    assert opt.defaults["weight_decay"] == 0.0
    assert opt.defaults["amsgrad"] is False
    print(f"Adam: betas={opt.defaults['betas']} eps={opt.defaults['eps']} "
          f"wd={opt.defaults['weight_decay']} amsgrad={opt.defaults['amsgrad']}")

    # ── One optimizer step runs ──────────────────────────────────────────────
    opt.zero_grad(set_to_none=True)
    logits = model(batch)
    loss = multi_sector_bce_loss(logits, labels)
    loss.backward()
    opt.step()
    with torch.no_grad():
        loss2 = multi_sector_bce_loss(model(batch), labels)
    print(f"one Adam step: loss {loss.item():.4f} -> {loss2.item():.4f}")

    # ── Majority vote still works on logits ──────────────────────────────────
    preds = majority_vote(logits.detach())
    assert preds.shape == (4,)
    print(f"majority_vote on batch: {preds.tolist()} vs labels {labels.long().tolist()}")

    print("\ntrain.py self-test PASSED.")


if __name__ == "__main__":
    _self_test()
