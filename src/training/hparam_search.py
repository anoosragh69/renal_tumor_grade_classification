"""
Task 14: Hyperparameter Tuning
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import itertools
import torch
import torch.nn as nn
from src.models.multimodal import MultimodalClassifier


def train_one_config(img, clin, labels, config):
    model = MultimodalClassifier(
        num_classes=config["num_classes"],
        dropout=config["dropout"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 1e-4)
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    out = model(img, clin)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_val = model(img, clin)
        val_loss = criterion(out_val, labels).item()
        pred = torch.argmax(out_val, dim=1)
        acc = (pred == labels).float().mean().item()

    params = sum(p.numel() for p in model.parameters())
    return {
        "config": config,
        "train_loss": loss.item(),
        "val_loss": val_loss,
        "val_acc": acc,
        "params": params,
    }


def grid_search():
    img = torch.randn(4, 1, 128, 128, 128)
    clin = torch.randn(4, 32)
    labels = torch.tensor([0, 1, 2, 3])

    configs = []
    for lr in [1e-4, 1e-3, 1e-2]:
        for dropout in [0.2, 0.3, 0.5]:
            configs.append({
                "lr": lr,
                "dropout": dropout,
                "num_classes": 4,
                "weight_decay": 1e-4,
            })

    print(f"Testing {len(configs)} configurations...\n")
    results = []
    for i, cfg in enumerate(configs):
        result = train_one_config(img, clin, labels, cfg)
        results.append(result)
        print(f"Config {i+1}: lr={cfg['lr']:.0e} dropout={cfg['dropout']} "
              f"-> loss={result['train_loss']:.4f} acc={result['val_acc']:.3f}")

    best = min(results, key=lambda x: x["val_loss"])
    print(f"\nBest config: lr={best['config']['lr']:.0e}, dropout={best['config']['dropout']}")
    print(f"Best val_loss: {best['val_loss']:.4f}, val_acc: {best['val_acc']:.3f}")

    return results, best


if __name__ == "__main__":
    results, best = grid_search()
    print("\nHyperparameter tuning complete!")
