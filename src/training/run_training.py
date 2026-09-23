"""
Step 7d: Full vViT training run (plan §4 Step 7.9 / M3)
=======================================================
200 epochs with Adam (paper hyperparams via ``build_paper_adam``), best
validation-accuracy checkpointing, TensorBoard logging (JSONL fallback when
tensorboard is unavailable).

Model/loss: VViT (Step 7a) + majority-vote fusion proxy for metrics
(Step 7b) + joint multi-sector BCE (Step 7c).

Outputs (git-ignored under results/):
    results/checkpoints/vvit_best.pt      best-val-accuracy state_dict + meta
    results/checkpoints/vvit_last.pt      final-epoch state_dict + meta
    results/metrics/train_metrics.jsonl   per-epoch metrics (always written)
    results/runs/<timestamp>/             TensorBoard events (if available)

Usage:
    python src/training/run_training.py                      # 200 epochs, defaults
    python src/training/run_training.py --epochs 2 --smoke   # quick smoke run
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch

from src.data.dataset import SECTOR_KEYS, get_dataloaders
from src.models.vvit import VViT
from src.models.fusion_baseline_vote import majority_vote
from src.training.train import multi_sector_bce_loss, build_paper_adam

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO_ROOT, "results")
CKPT_DIR = os.path.join(RESULTS, "checkpoints")
METRICS_DIR = os.path.join(RESULTS, "metrics")
RUNS_DIR = os.path.join(RESULTS, "runs")


def get_writer(run_name: str):
    """TensorBoard writer if available, else None (JSONL fallback is always on)."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = os.path.join(RUNS_DIR, run_name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir)
    except Exception as e:  # no tensorboard package / CUDA-free envs
        print(f"TensorBoard unavailable ({e}); JSONL metrics only.", flush=True)
        return None


def batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: VViT, loader, device: torch.device) -> dict:
    """Mean val BCE + majority-vote accuracy over the loader."""
    model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for batch in loader:
        labels = batch["label"].to(device)
        batch = batch_to_device(batch, device)
        logits = model(batch)
        loss = multi_sector_bce_loss(logits, labels)
        preds = majority_vote(logits)
        total_loss += loss.item() * labels.shape[0]
        total_correct += int((preds == labels.long()).sum().item())
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
        "model_config": {
            "embed_dim": model.embed_dim, "depth": model.depth,
            "num_heads": model.num_heads, "mlp_dim": model.mlp_dim,
        },
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)


def train(args) -> dict:
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cpu" if args.device == "auto" else args.device)
    )
    print(f"device: {device}", flush=True)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    loaders = get_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers,
                              augment=not args.no_aug, seed=args.seed)
    n_train, n_val = len(loaders["train"].dataset), len(loaders["val"].dataset)
    print(f"train slices: {n_train} | val slices: {n_val}", flush=True)

    model = VViT().to(device)
    optimizer = build_paper_adam(model.parameters(), lr=args.lr)

    run_name = time.strftime("%Y%m%d-%H%M%S")
    writer = get_writer(run_name)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    jsonl_path = os.path.join(METRICS_DIR, "train_metrics.jsonl")
    metrics_f = open(jsonl_path, "a", encoding="utf-8")

    config = vars(args) | {"run_name": run_name, "device": str(device),
                           "train_slices": n_train, "val_slices": n_val}
    metrics_f.write(json.dumps({"type": "config", **config}) + "\n")
    metrics_f.flush()

    best_val_acc = -1.0
    best_epoch = -1
    epochs = args.epochs if not args.smoke else min(args.epochs, 2)

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss, ep_n = 0.0, 0
        for batch in loaders["train"]:
            labels = batch["label"].to(device)
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = multi_sector_bce_loss(logits, labels)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * labels.shape[0]
            ep_n += labels.shape[0]
        train_loss = ep_loss / max(ep_n, 1)

        val = evaluate(model, loaders["val"], device)
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
        print(f"epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val['val_loss']:.4f} | val_acc {val['val_acc']:.4f} | {elapsed:.1f}s",
              flush=True)

        if val["val_acc"] > best_val_acc:
            best_val_acc = val["val_acc"]
            best_epoch = epoch
            save_checkpoint(model, optimizer, epoch, row,
                            os.path.join(CKPT_DIR, "vvit_best.pt"),
                            extra={"run_name": run_name})

    save_checkpoint(model, optimizer, epochs, row,
                    os.path.join(CKPT_DIR, "vvit_last.pt"),
                    extra={"run_name": run_name})
    if writer is not None:
        writer.close()
    metrics_f.close()

    summary = {"run_name": run_name, "best_epoch": best_epoch,
               "best_val_acc": best_val_acc, **config}
    with open(os.path.join(METRICS_DIR, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBest val_acc {best_val_acc:.4f} @ epoch {best_epoch}. "
          f"Checkpoints -> {CKPT_DIR}", flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Step 7d: full vViT training (200 epochs, best-val checkpoint)")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Adam lr — paper does not specify; default 1e-3 (documented choice)")
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-aug", action="store_true", help="Disable train augmentations")
    p.add_argument("--smoke", action="store_true", help="Cap at 2 epochs for a smoke run")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
