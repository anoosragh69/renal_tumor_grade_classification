"""
Step 8: Train & evaluate timm baselines under the Step 7d / 9a protocol
========================================================================
For each backbone in ``BASELINE_MODELS`` (vit / convnext / resnext):

  train:  same splits, same train-only paper augmentations, same Adam
          (build_paper_adam, lr default 1e-3), BCE with logits on the single
          binary head, 200 epochs, best-val-accuracy checkpointing,
          JSONL (+ TensorBoard when available) logging.
  eval:   load best-val checkpoint -> test split, image-level binary metrics
          with bootstrap 95% CI + patient-level aggregation (majority vote /
          mean prob) — mirrors run_evaluation.py so M4 DeLong/McNemar can
          compare against the vViT predictions directly.

Outputs (git-ignored under results/):
    results/checkpoints/<tag>_best.pt / <tag>_last.pt
    results/metrics/baseline_<tag>_metrics.jsonl      per-epoch training metrics
    results/metrics/baseline_<tag>_test_metrics.json  test metrics incl. CIs
    results/metrics/baseline_<tag>_test_predictions.csv

Usage:
    python src/training/run_baselines.py                       # all 3, 200 epochs
    python src/training/run_baselines.py --models vit --smoke  # quick smoke
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.data.dataset import RenalTumorSectorDataset, get_dataloaders
from src.training.baselines import BASELINE_MODELS, TimmBaseline
from src.training.evaluation import (
    aggregate_patient, binary_metrics, bootstrap_ci, format_metric_with_ci,
)
from src.training.run_training import (
    CKPT_DIR, METRICS_DIR, batch_to_device, get_writer,
)
from src.training.train import build_paper_adam
from src.training.run_evaluation import POINT_KEYS


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Mean val BCE + accuracy (sigmoid >= 0.5) over the loader."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        labels = batch["label"].to(device)
        logits = model(batch_to_device(batch, device))
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        total_loss += loss.item() * labels.shape[0]
        total_correct += int(((logits >= 0).long() == labels.long()).sum().item())
        total_n += labels.shape[0]
    if total_n == 0:
        return {"val_loss": float("nan"), "val_acc": float("nan")}
    return {"val_loss": total_loss / total_n, "val_acc": total_correct / total_n}


def save_checkpoint(model, optimizer, epoch, metrics, path, extra=None):
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "model_config": {"model_name": model.model_name},
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def train_one(tag, args, loaders) -> dict:
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TimmBaseline(tag, pretrained=not args.scratch).to(device)
    optimizer = build_paper_adam(model.parameters(), lr=args.lr)

    run_name = f"baseline-{tag}-" + time.strftime("%Y%m%d-%H%M%S")
    writer = get_writer(run_name)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    jsonl_path = os.path.join(METRICS_DIR, f"baseline_{tag}_metrics.jsonl")
    metrics_f = open(jsonl_path, "a", encoding="utf-8")

    config = vars(args) | {"run_name": run_name, "device": str(device),
                           "model_tag": tag, "timm_name": BASELINE_MODELS[tag],
                           "train_slices": len(loaders["train"].dataset),
                           "val_slices": len(loaders["val"].dataset)}
    metrics_f.write(json.dumps({"type": "config", **config}) + "\n")
    metrics_f.flush()
    print(f"\n### {tag} ({BASELINE_MODELS[tag]}) on {device} — run {run_name}",
          flush=True)

    best_val_acc, best_epoch = -1.0, -1
    epochs = args.epochs if not args.smoke else min(args.epochs, 2)
    train_cap = 10 if args.smoke else None   # smoke: cap batches/epoch (fast verify)
    val_cap = 5 if args.smoke else None
    row = {}
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss, ep_n = 0.0, 0
        for bi, batch in enumerate(loaders["train"]):
            if train_cap is not None and bi >= train_cap:
                break
            labels = batch["label"].to(device)
            logits = model(batch_to_device(batch, device))
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * labels.shape[0]
            ep_n += labels.shape[0]
        train_loss = ep_loss / max(ep_n, 1)

        val = evaluate(model, loaders["val"], device, max_batches=val_cap)
        elapsed = time.time() - t0
        row = {"type": "epoch", "run": run_name, "epoch": epoch,
               "train_loss": train_loss, "val_loss": val["val_loss"],
               "val_acc": val["val_acc"], "lr": args.lr, "sec": round(elapsed, 2)}
        metrics_f.write(json.dumps(row) + "\n")
        metrics_f.flush()
        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val["val_loss"], epoch)
            writer.add_scalar("acc/val", val["val_acc"], epoch)
        print(f"  [{tag}] epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val['val_loss']:.4f} | val_acc {val['val_acc']:.4f} | "
              f"{elapsed:.1f}s", flush=True)

        if val["val_acc"] > best_val_acc:
            best_val_acc = val["val_acc"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, row,
                            os.path.join(CKPT_DIR, f"{tag}_best.pt"),
                            extra={"run_name": run_name})

    save_checkpoint(model, optimizer, epochs, row,
                    os.path.join(CKPT_DIR, f"{tag}_last.pt"),
                    extra={"run_name": run_name})
    if writer is not None:
        writer.close()
    metrics_f.close()

    summary = {"run_name": run_name, "model_tag": tag,
               "best_epoch": best_epoch, "best_val_acc": best_val_acc, **config}
    with open(os.path.join(METRICS_DIR, f"baseline_{tag}_train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [{tag}] best val_acc {best_val_acc:.4f} @ epoch {best_epoch}", flush=True)
    return summary


@torch.no_grad()
def predict_split(model, loader, device, max_batches=None):
    """Per-slice inference -> DataFrame (label, pred, prob, patient_id)."""
    model.eval()
    rows = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        labels = batch["label"].to(device)
        probs = torch.sigmoid(model(batch_to_device(batch, device)))
        preds = (probs >= 0.5).long()
        for i in range(labels.shape[0]):
            rows.append({
                "patient_id": batch["patient_id"][i],
                "slice_idx": int(batch["slice_idx"][i]),
                "label": int(labels[i]),
                "pred": int(preds[i]),
                "prob": float(probs[i]),
            })
    return pd.DataFrame(rows)


def summarize(y_true, y_pred, y_score, n_boot, seed):
    m = binary_metrics(y_true, y_pred, y_score)
    ci = bootstrap_ci(y_true, y_pred, y_score, n_boot=n_boot, seed=seed)
    return {k: {"value": m.get(k),
                "ci95": list(ci[k]) if k in ci else None} for k in POINT_KEYS}


def eval_one(tag, args, device):
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TimmBaseline(ckpt["model_config"].get("model_name", tag),
                         pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    ds = RenalTumorSectorDataset(split="test", augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size,
                                         shuffle=False)
    eval_cap = 10 if getattr(args, "smoke", False) else None
    print(f"\n[{tag}] test eval: {len(ds.patient_ids)} patients, {len(ds)} slices "
          f"(ckpt epoch {ckpt.get('epoch')}, val_acc "
          f"{ckpt.get('metrics', {}).get('val_acc')})"
          + (f" [smoke: capped at {eval_cap} batches]" if eval_cap else ""), flush=True)

    df = predict_split(model, loader, device, max_batches=eval_cap)
    pred_path = os.path.join(METRICS_DIR, f"baseline_{tag}_test_predictions.csv")
    df.to_csv(pred_path, index=False)

    y = df["label"].to_numpy()
    image = summarize(y, df["pred"].to_numpy(), df["prob"].to_numpy(),
                      args.n_boot, args.seed)

    agg_vote = aggregate_patient(df["pred"], df["prob"], df["patient_id"], mode="vote")
    agg_mean = aggregate_patient(df["pred"], df["prob"], df["patient_id"], mode="mean")
    pat_ids = sorted(agg_vote)
    y_pat = np.array([int(df[df["patient_id"] == pid]["label"].iloc[0]) for pid in pat_ids])
    pred_vote = np.array([agg_vote[pid][0] for pid in pat_ids])
    score_mean = np.array([agg_mean[pid][1] for pid in pat_ids])
    patient = summarize(y_pat, pred_vote, score_mean, args.n_boot, args.seed)

    out = {
        "model_tag": tag, "timm_name": BASELINE_MODELS[tag],
        "checkpoint": os.path.relpath(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": "test", "n_slices": int(len(df)),
        "patients": {"patients": int(len(y_pat)),
                     "low": int((y_pat == 0).sum()),
                     "high": int((y_pat == 1).sum())},
        "image_level": image,
        "patient_level": patient,
        "notes": [
            "single binary head: prob=sigmoid(logit), pred=prob>=0.5",
            "patient-level: majority vote labels, mean prob for AUC",
            "timm ImageNet-pretrained, in_chans=1 stem fold, 128->224 bilinear resize",
        ],
    }
    out_path = os.path.join(METRICS_DIR, f"baseline_{tag}_test_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"  [{tag}] image-level:", flush=True)
    for k in POINT_KEYS:
        print(f"    {format_metric_with_ci(binary_metrics(y, df['pred'].to_numpy(), df['prob'].to_numpy()), bootstrap_ci(y, df['pred'].to_numpy(), df['prob'].to_numpy(), n_boot=args.n_boot, seed=args.seed), k)}")
    print(f"  [{tag}] patient-level:", flush=True)
    for k in POINT_KEYS:
        print(f"    {format_metric_with_ci(binary_metrics(y_pat, pred_vote, score_mean), bootstrap_ci(y_pat, pred_vote, score_mean, n_boot=args.n_boot, seed=args.seed), k)}")
    print(f"  Saved -> {out_path} / {pred_path}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 8: timm baselines, same protocol as Step 7d/9a")
    p.add_argument("--models", nargs="+", default=list(BASELINE_MODELS),
                   choices=list(BASELINE_MODELS))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="same default as run_training.py (documented choice)")
    p.add_argument("--scratch", action="store_true",
                   help="random init instead of ImageNet-pretrained (deviation toggle)")
    p.add_argument("--eval-only", action="store_true",
                   help="skip training, run test eval on existing *_best.pt")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs per model, no full test eval (CI sanity only)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    if not args.eval_only:
        loaders = get_dataloaders(batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  augment=True, seed=args.seed)
        n_train = len(loaders["train"].dataset)
        print(f"train slices: {n_train} | val slices: {len(loaders['val'].dataset)}",
              flush=True)
        train_batch = min(args.batch_size, 256) if not args.smoke else 16
        if args.smoke:
            # cap smoke throughput: smaller batches keep it quick on 6GB GPUs
            loaders = get_dataloaders(batch_size=train_batch,
                                      num_workers=args.num_workers,
                                      augment=True, seed=args.seed)
        summaries = {tag: train_one(tag, args, loaders) for tag in args.models}
        print("\nTraining summaries:", flush=True)
        for tag, s in summaries.items():
            print(f"  {tag:8s}: best_val_acc {s['best_val_acc']:.4f} @ ep {s['best_epoch']}",
                  flush=True)
    n_boot = 50 if args.smoke else args.n_boot
    for tag in args.models:
        eval_args = argparse.Namespace(**{**vars(args), "n_boot": n_boot})
        eval_one(tag, eval_args, device)


if __name__ == "__main__":
    main()
