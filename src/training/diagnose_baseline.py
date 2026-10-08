"""
Diagnose timm baseline non-learning (M4 blocker)
=================================================
Full runs pinned train_loss at ln(2)≈0.693 with val_acc at the majority
class — the backbones only fit their bias. This probe isolates the cause on
a FIXED batch of 16 real train slices (no augmentation):

  1. image tensor stats (range/scale feeding the pretrained backbones)
  2. grad norms (head vs backbone) + logit spread at init
  3. 60 Adam steps at lr 1e-3 (full finetune) vs 1e-4 vs head-only

A learnable config drives loss on the fixed batch well below ln(2)
(memorisation of 16 samples is easy) — that config is the training fix.

Usage:
    python src/training/diagnose_baseline.py                # vit + resnext
    python src/training/diagnose_baseline.py --models vit convnext resnext
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F

from src.data.dataset import RenalTumorSectorDataset
from src.training.baselines import BASELINE_MODELS, TimmBaseline
from src.training.run_training import batch_to_device
from src.training.train import build_paper_adam

LN2 = float(torch.log(torch.tensor(2.0)))


def head_params(model):
    head = model.backbone.get_classifier()
    ids = {id(p) for p in head.parameters()}
    head_p = [p for p in model.parameters() if id(p) in ids]
    body_p = [p for p in model.parameters() if id(p) not in ids]
    return head_p, body_p


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def run_config(tag, model, batch, lr, freeze_body, steps):
    _, body_p = head_params(model)
    if freeze_body:
        for p in body_p:
            p.requires_grad = False
    head_p, body_p = head_params(model)
    trainable = head_p + [p for p in body_p if p.requires_grad]
    opt = build_paper_adam(trainable, lr=lr)

    print(f"  [{tag}] lr={lr:g} freeze_body={freeze_body} "
          f"trainable={sum(p.numel() for p in trainable):,} params")
    t0 = torch.cuda.synchronize() if torch.cuda.is_available() else None
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(batch)
        loss = F.binary_cross_entropy_with_logits(logits, batch["label"])
        loss.backward()
        hgn = grad_norm(head_p)
        bgn = grad_norm([p for p in body_p if p.requires_grad])
        opt.step()
        if step == 1 or step % 10 == 0 or step == steps:
            std = float(logits.detach().std())
            print(f"    step {step:3d}  loss {loss.item():.4f}  "
                  f"logit_std {std:.4f}  |g| head {hgn:.3e} body {bgn:.3e}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return loss.item()


def _feature_stats(model, batch):
    """Diversity of backbone features across the batch (mean-pairwise cosine
    near 1.0 = collapsed/constant representation)."""
    img = batch["image"]
    if img.shape[-1] != 224:   # production path resizes to 224
        img = torch.nn.functional.interpolate(
            img, size=(224, 224), mode="bilinear", align_corners=False)
    with torch.no_grad():
        feats = model.backbone.forward_features(img)
        if feats.dim() > 2:
            feats = model.backbone.forward_head(feats, pre_logits=True)
        feats = feats.reshape(feats.shape[0], -1).float()
        per_dim_std = float(feats.std(dim=0).mean())
        centered = feats - feats.mean(dim=0, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(
            centered.unsqueeze(1), centered.unsqueeze(0), dim=2)
        offdiag = cos[~torch.eye(len(feats), dtype=torch.bool, device=cos.device)]
        eff_rank = torch.linalg.matrix_rank(
            centered.T @ centered, tol=1e-6 * centered.shape[0]).item()
    return per_dim_std, float(offdiag.mean()), int(eff_rank), feats.shape[1]


def feature_probe(models, batch, steps=60):
    """Does the backbone representation collapse under training?

    Trains a fixed mixed-label batch for ``steps`` at lr 1e-3 vs 1e-4, then
    measures feature diversity. Collapse signature: mean pairwise cosine -> 1
    (all samples share one feature vector), logit_std -> 0, loss -> ln2.
    """
    labels = batch["label"]
    pos_rate = float(labels.mean())
    entropy = float(-(pos_rate * torch.log(torch.tensor(pos_rate + 1e-12))
                      + (1 - pos_rate) * torch.log(torch.tensor(1 - pos_rate + 1e-12))))
    print(f"\n=== feature probe (mixed labels, pos-rate {pos_rate:.3f}, "
          f"entropy {entropy:.4f}) ===")
    for name in models:
        for tag, cfg in ((f"{name}@init", None),
                         (f"{name}@1e-3", dict(lr=1e-3, freeze_body=False)),
                         (f"{name}@1e-4", dict(lr=1e-4, freeze_body=False))):
            model = TimmBaseline(name, pretrained=True).to(batch["image"].device)
            model.train()   # match run_baselines training mode (BN stats)
            if cfg is not None:
                run_config(tag, model, batch, steps=steps, **cfg)
            std, cos, rank, dim = _feature_stats(model, batch)
            # collapse signature: features shrink to noise (std -> 0, rank -> ~0);
            # cosine is uninformative once magnitudes vanish
            collapsed = (cos > 0.95 or std < 0.01)
            print(f"  {tag:16s}: feat_std/dim {std:.4f}  "
                  f"mean-pairwise-cos {cos:.3f}  "
                  f"eff_rank {rank}/{dim}"
                  + ("  <-- COLLAPSED" if collapsed else ""))
            del model
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["vit", "resnext"],
                   choices=list(BASELINE_MODELS))
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--feature-probe", action="store_true",
                   help="feature-diversity probe only (skip optimisation runs)")
    p.add_argument("--mixed-batch", action="store_true",
                   help="sample the fixed batch across patients/labels instead "
                        "of the first N (all one class)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    ds = RenalTumorSectorDataset(split="train", augment=False)
    if args.mixed_batch or args.feature_probe:
        # alternate low/high patients so discrimination is actually tested
        from collections import defaultdict
        by_label = defaultdict(list)
        for i, (pid, _) in enumerate(ds.index):
            by_label[ds.clinical[pid][3]].append(i)
        picks = []
        while len(picks) < args.batch_size:
            for lab in (0, 1):
                if by_label[lab]:
                    picks.append(by_label[lab].pop(0))
                if len(picks) >= args.batch_size:
                    break
        batch = torch.utils.data.default_collate([ds[i] for i in picks])
    else:
        batch = torch.utils.data.default_collate(
            [ds[i] for i in range(args.batch_size)])
    batch = batch_to_device(batch, device)
    labels = batch["label"]
    img = batch["image"]
    print(f"device={device}  fixed batch: {tuple(img.shape)}  "
          f"img min {img.min():.3f} max {img.max():.3f} "
          f"mean {img.mean():.3f} std {img.std():.3f}  "
          f"labels {labels.tolist()}  pos-rate {labels.mean():.3f}  "
          f"(ln2={LN2:.4f})")

    if args.feature_probe:
        feature_probe(args.models, batch)
        return

    for name in args.models:
        print(f"\n=== {name} ({BASELINE_MODELS[name]}) ===")
        configs = [
            (f"{name}@1e-3", dict(lr=1e-3, freeze_body=False)),
            (f"{name}@1e-4", dict(lr=1e-4, freeze_body=False)),
            (f"{name}@head-only", dict(lr=1e-3, freeze_body=True)),
        ]
        for tag, cfg in configs:
            model = TimmBaseline(name, pretrained=True).to(device)
            model.train()
            final = run_config(tag, model, batch, steps=args.steps, **cfg)
            verdict = "LEARNS" if final < 0.5 * LN2 else \
                      ("partial" if final < LN2 - 0.02 else "STUCK at ln2")
            print(f"    -> final loss {final:.4f}  [{verdict}]")
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
