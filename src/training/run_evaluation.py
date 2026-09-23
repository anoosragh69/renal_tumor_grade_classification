"""
Step 9a: Full test-set evaluation of the trained vViT (plan §4 Step 9.1–9.2)
============================================================================
Loads a checkpoint (default: best-val from Step 7d), runs inference over the
test split, and computes:

  image-level (per-slice)
    accuracy, sensitivity, specificity, PPV, NPV, F1, Cohen's kappa,
    AUC-ROC, log loss — each with bootstrap 95% CI (1000x resamples)
  per-sector-head accuracy table (basis for the paper's Table 2)
  patient-level
    majority vote for the hard label, mean probability for AUC
    (plan §4 Step 9.2), same metrics + bootstrap CI

Score convention (documented choice): a slice's continuous score is the mean
sigmoid probability across the 6 sector heads; the hard slice prediction is
the Step 7b majority vote (class-token tie-break).

Outputs (git-ignored under results/):
    results/metrics/test_metrics.json       full metrics incl. CIs
    results/metrics/test_predictions.csv    per-slice patient_id, slice_idx,
                                            label, majority pred, mean prob,
                                            per-head preds

Usage:
    python src/training/run_evaluation.py                     # best ckpt, test split
    python src/training/run_evaluation.py --checkpoint results/checkpoints/vvit_last.pt --split val
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import torch

from src.data.dataset import get_dataloaders
from src.models.vvit import VViT, TOKEN_ORDER
from src.models.fusion_baseline_vote import majority_vote
from src.training.evaluation import (
    binary_metrics, bootstrap_ci, aggregate_patient, format_metric_with_ci,
)
from src.training.run_training import batch_to_device, RESULTS, METRICS_DIR

DEFAULT_CKPT = os.path.join(RESULTS, "checkpoints", "vvit_best.pt")

POINT_KEYS = ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1",
              "kappa", "auroc", "log_loss"]


@torch.no_grad()
def predict_split(model, loader, device):
    """Per-slice inference -> DataFrame with preds/probs per head."""
    model.eval()
    rows = []
    for batch in loader:
        labels = batch["label"].to(device)
        torch_batch = batch_to_device(batch, device)
        logits = model(torch_batch)                       # (B, 6)
        probs = torch.sigmoid(logits)
        voted = majority_vote(logits)                     # (B,)
        mean_prob = probs.mean(dim=1)                     # (B,)
        head_preds = (probs >= 0.5)
        for i in range(labels.shape[0]):
            row = {
                "patient_id": batch["patient_id"][i],
                "slice_idx": int(batch["slice_idx"][i]),
                "label": int(labels[i]),
                "vote_pred": int(voted[i]),
                "mean_prob": float(mean_prob[i]),
            }
            for h, tok in enumerate(TOKEN_ORDER):
                row[f"pred_{tok}"] = int(head_preds[i, h])
                row[f"prob_{tok}"] = float(probs[i, h])
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(y_true, y_pred, y_score, n_boot):
    m = binary_metrics(y_true, y_pred, y_score)
    ci = bootstrap_ci(y_true, y_pred, y_score, n_boot=n_boot)
    return {k: {"value": m.get(k),
                "ci95": list(ci[k]) if k in ci else None} for k in POINT_KEYS}


def main():
    p = argparse.ArgumentParser(description="Step 9a: test-set evaluation of vViT")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = VViT(**cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch')}, "
          f"val_acc {ckpt.get('metrics', {}).get('val_acc')}) on {device}", flush=True)

    # Deterministic loader: no augmentation, no shuffle
    from src.data.dataset import RenalTumorSectorDataset
    ds = RenalTumorSectorDataset(split=args.split, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"{args.split}: {len(ds.patient_ids)} patients, {len(ds)} slices", flush=True)

    df = predict_split(model, loader, device)
    os.makedirs(METRICS_DIR, exist_ok=True)
    pred_path = os.path.join(METRICS_DIR, f"{args.split}_predictions.csv")
    df.to_csv(pred_path, index=False)

    # ── Image-level (per-slice) ───────────────────────────────────────────────
    y = df["label"].to_numpy()
    image = summarize(y, df["vote_pred"].to_numpy(), df["mean_prob"].to_numpy(),
                      args.n_boot)

    # ── Per-sector-head table (paper Table 2 analogue) ────────────────────────
    heads = {}
    for tok in TOKEN_ORDER:
        hp = df[f"pred_{tok}"].to_numpy()
        heads[tok] = {
            "accuracy": float((hp == y).mean()),
            "auroc": binary_metrics(y, hp, df[f"prob_{tok}"].to_numpy())["auroc"],
        }

    # ── Patient-level (plan Step 9.2) ─────────────────────────────────────────
    agg_vote = aggregate_patient(df["vote_pred"], df["mean_prob"],
                                 df["patient_id"], mode="vote")
    agg_mean = aggregate_patient(df["vote_pred"], df["mean_prob"],
                                 df["patient_id"], mode="mean")
    pat_ids = sorted(agg_vote)
    y_pat = np.array([int(df[df["patient_id"] == pid]["label"].iloc[0]) for pid in pat_ids])
    pred_vote = np.array([agg_vote[pid][0] for pid in pat_ids])
    score_mean = np.array([agg_mean[pid][1] for pid in pat_ids])
    patient = summarize(y_pat, pred_vote, score_mean, args.n_boot)
    patient_n = {"patients": int(len(y_pat)),
                 "low": int((y_pat == 0).sum()), "high": int((y_pat == 1).sum())}

    out = {
        "checkpoint": os.path.relpath(args.checkpoint, RESULTS),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split,
        "n_slices": int(len(df)),
        "patients": patient_n,
        "image_level": image,
        "per_head": heads,
        "patient_level": patient,
        "notes": [
            "slice score = mean sigmoid prob over 6 heads; hard pred = majority vote (class-token tie-break)",
            "patient-level: majority vote labels, mean prob for AUC",
        ],
    }
    out_path = os.path.join(METRICS_DIR, f"{args.split}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # ── Pretty print ──────────────────────────────────────────────────────────
    print(f"\n=== Image-level ({args.split}, n={len(df)} slices) ===", flush=True)
    img_pt = binary_metrics(y, df["vote_pred"].to_numpy(), df["mean_prob"].to_numpy())
    img_ci = bootstrap_ci(y, df["vote_pred"].to_numpy(), df["mean_prob"].to_numpy(),
                          n_boot=args.n_boot, seed=args.seed)
    for k in POINT_KEYS:
        print(f"  {format_metric_with_ci(img_pt, img_ci, k)}")
    print(f"\n=== Per-head accuracy ===", flush=True)
    for tok, h in heads.items():
        print(f"  {tok:12s} acc={h['accuracy']:.3f}  auroc={h['auroc']:.3f}")
    print(f"\n=== Patient-level (n={len(y_pat)} = {patient_n['low']} low + {patient_n['high']} high) ===",
          flush=True)
    pat_pt = binary_metrics(y_pat, pred_vote, score_mean)
    pat_ci = bootstrap_ci(y_pat, pred_vote, score_mean, n_boot=args.n_boot, seed=args.seed)
    for k in POINT_KEYS:
        print(f"  {format_metric_with_ci(pat_pt, pat_ci, k)}")
    print(f"\nSaved -> {out_path}")
    print(f"Saved -> {pred_path}")


if __name__ == "__main__":
    main()
