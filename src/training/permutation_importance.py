"""Patient-wise permutation importance for a trained vViT checkpoint."""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import auc, roc_curve

from src.data.dataset import RenalTumorSectorDataset, SECTOR_KEYS, get_paths
from src.models.fusion_baseline_vote import majority_vote
from src.models.vvit import VViT
from src.training.evaluation import aggregate_patient, binary_metrics, pairwise_mann_whitney
from src.training.run_training import CKPT_DIR, METRICS_DIR, RESULTS


DEFAULT_CHECKPOINT = os.path.join(CKPT_DIR, "vvit_paper_seed42_best.pt")
CLINICAL_SECTORS = {"demographic", "comorbidity", "habit"}


def load_test_tensors(dataset, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )
    sectors = {key: [] for key in SECTOR_KEYS}
    labels, patient_ids, slice_indices = [], [], []
    for batch in loader:
        for key in SECTOR_KEYS:
            sectors[key].append(batch[key].cpu())
        labels.append(batch["label"].cpu())
        patient_ids.extend(batch["patient_id"])
        slice_indices.append(batch["slice_idx"].cpu())
    tensors = {key: torch.cat(values) for key, values in sectors.items()}
    labels = torch.cat(labels).numpy().astype(int)
    slice_indices = torch.cat(slice_indices).numpy().astype(int)
    patient_ids = np.asarray(patient_ids, dtype=str)
    groups = {}
    for pid in np.unique(patient_ids):
        idx = np.flatnonzero(patient_ids == pid)
        groups[pid] = idx[np.argsort(slice_indices[idx])]
    return tensors, labels, patient_ids, groups


def predict(model, tensors, labels, patient_ids, device, batch_size):
    all_preds, all_scores = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            batch = {key: value[start:end].to(device, non_blocking=True)
                     for key, value in tensors.items()}
            logits = model(batch)
            probs = torch.sigmoid(logits.float())
            all_preds.append(majority_vote(logits.float()).cpu().numpy())
            all_scores.append(probs.mean(dim=1).cpu().numpy())
    preds = np.concatenate(all_preds).astype(int)
    scores = np.concatenate(all_scores)
    patient_vote = aggregate_patient(preds, scores, patient_ids, mode="vote")
    patient_mean = aggregate_patient(preds, scores, patient_ids, mode="mean")
    pids = sorted(patient_vote)
    patient_y = pd.Series(labels, index=patient_ids).groupby(level=0).first().reindex(pids).to_numpy(int)
    patient_pred = np.array([patient_vote[pid][0] for pid in pids])
    patient_score = np.array([patient_mean[pid][1] for pid in pids])
    metrics = {
        "image_accuracy": float(np.mean(preds == labels)),
        "patient_accuracy": float(np.mean(patient_pred == patient_y)),
        "image_auroc": binary_metrics(labels, preds, scores)["auroc"],
        "patient_auroc": binary_metrics(patient_y, patient_pred, patient_score)["auroc"],
    }
    return metrics, preds, scores, patient_y, patient_pred, patient_score, pids


def permute_patient_profiles(tensor, patient_ids, groups, sector, rng):
    """Reassign whole patient profiles, aligning slice-wise profiles by rank."""
    pids = np.asarray(sorted(groups))
    order = rng.permutation(pids)
    shift = int(rng.integers(1, len(order)))
    donors = np.roll(order, shift)
    result = tensor.clone()
    for target_pid, donor_pid in zip(order, donors):
        target_rows = torch.as_tensor(groups[target_pid], dtype=torch.long)
        donor_rows = torch.as_tensor(groups[donor_pid], dtype=torch.long)
        if sector in CLINICAL_SECTORS:
            values = tensor.index_select(0, donor_rows[:1]).expand(
                (len(target_rows),) + tuple(tensor.shape[1:])
            )
        else:
            donor_profile = tensor.index_select(0, donor_rows)
            ranks = torch.linspace(0, len(donor_rows) - 1, steps=len(target_rows)).round().long()
            values = donor_profile.index_select(0, ranks)
        result.index_copy_(0, target_rows, values)
    return result


def save_figures(rows, labels, image_scores, patient_y, patient_scores, output_tag, figure_dir):
    sectors = list(SECTOR_KEYS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "image_drop", "Image-level accuracy drop"),
        (axes[1], "patient_drop", "Patient-level accuracy drop"),
    ):
        samples = [[row[metric] for row in rows if row["sector"] == sector] for sector in sectors]
        ax.boxplot(samples, tick_labels=sectors, showfliers=False)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Baseline accuracy − permuted accuracy")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    importance_path = os.path.join(figure_dir, f"vvit_permutation_importance_{output_tag}.png")
    fig.savefig(importance_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, y, score, title in (
        (axes[0], labels, image_scores, "Image-level ROC"),
        (axes[1], patient_y, patient_scores, "Patient-level ROC"),
    ):
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, label=f"vViT (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate",
               ylabel="True positive rate", title=title)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
    roc_path = os.path.join(figure_dir, f"vvit_roc_{output_tag}.png")
    fig.savefig(roc_path, dpi=180)
    plt.close(fig)
    return importance_path, roc_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Patient-wise vViT permutation importance")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-tag", default="seed42")
    args = parser.parse_args(argv)
    if args.n_permutations < 1:
        parser.error("--n-permutations must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.output_tag):
        parser.error("--output-tag may contain only letters, digits, dot, underscore, and hyphen")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    preprocessing = checkpoint.get("preprocessing")
    if not preprocessing or "radiomic_mean" not in preprocessing:
        raise ValueError("Checkpoint predates train-only radiomic scaling; use a current checkpoint")
    paths = get_paths(
        splits_file=checkpoint["data_paths"]["splits"],
        top16_file=checkpoint["data_paths"]["top16"],
        radiomics_file=checkpoint["data_paths"]["radiomics"],
    )
    dataset = RenalTumorSectorDataset(
        split="test", augment=False, paths=paths, preprocessing=preprocessing
    )
    tensors, labels, patient_ids, groups = load_test_tensors(
        dataset, args.batch_size, args.num_workers
    )
    model = VViT(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])

    baseline, baseline_preds, baseline_scores, patient_y, patient_pred, patient_scores, pids = predict(
        model, tensors, labels, patient_ids, device, args.batch_size
    )
    expected = {
        "image_accuracy": baseline["image_accuracy"],
        "patient_accuracy": baseline["patient_accuracy"],
        "image_auroc": baseline["image_auroc"],
        "patient_auroc": baseline["patient_auroc"],
    }
    saved_metrics_path = os.path.join(METRICS_DIR, f"test_metrics_{checkpoint.get('run_tag')}.json")
    if os.path.exists(saved_metrics_path):
        saved = json.load(open(saved_metrics_path, encoding="utf-8"))
        for key in expected:
            level, metric = key.split("_")[0], key.split("_")[1]
            saved_value = saved[f"{level}_level"][metric]["value"]
            if abs(expected[key] - saved_value) > 1e-6:
                raise ValueError(
                    f"Permutation baseline {key}={expected[key]:.8f} does not match "
                    f"saved evaluation {saved_value:.8f}"
                )

    rng = np.random.default_rng(args.seed)
    rows = []
    total = len(SECTOR_KEYS) * args.n_permutations
    done = 0
    start_time = time.time()
    for sector in SECTOR_KEYS:
        for repeat in range(args.n_permutations):
            permuted_sector = permute_patient_profiles(
                tensors[sector], patient_ids, groups, sector, rng
            )
            permuted = dict(tensors)
            permuted[sector] = permuted_sector
            metrics, *_ = predict(model, permuted, labels, patient_ids, device, args.batch_size)
            rows.append({
                "sector": sector,
                "repeat": repeat,
                "image_accuracy": metrics["image_accuracy"],
                "patient_accuracy": metrics["patient_accuracy"],
                "image_drop": baseline["image_accuracy"] - metrics["image_accuracy"],
                "patient_drop": baseline["patient_accuracy"] - metrics["patient_accuracy"],
            })
            done += 1
            if done % 10 == 0 or done == total:
                print(f"permutations {done}/{total} ({time.time() - start_time:.1f}s)", flush=True)
            del permuted_sector, permuted

    os.makedirs(METRICS_DIR, exist_ok=True)
    figure_dir = os.path.join(RESULTS, "figures")
    os.makedirs(figure_dir, exist_ok=True)
    csv_path = os.path.join(METRICS_DIR, f"vvit_permutation_importance_{args.output_tag}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    image_pairs = pairwise_mann_whitney({
        sector: [row["image_drop"] for row in rows if row["sector"] == sector]
        for sector in SECTOR_KEYS
    })
    patient_pairs = pairwise_mann_whitney({
        sector: [row["patient_drop"] for row in rows if row["sector"] == sector]
        for sector in SECTOR_KEYS
    })
    summary = {
        "checkpoint": os.path.relpath(args.checkpoint, RESULTS),
        "run_tag": checkpoint.get("run_tag"),
        "n_patients": len(pids),
        "n_slices": len(labels),
        "n_permutations": args.n_permutations,
        "seed": args.seed,
        "permutation_alignment": "patient profiles reassigned without replacement; slice-wise profiles aligned by sorted slice rank and linearly spaced nearest ranks",
        "baseline": baseline,
        "importance": {
            sector: {
                metric: {
                    "mean_drop": float(np.mean([row[metric] for row in rows if row["sector"] == sector])),
                    "std_drop": float(np.std([row[metric] for row in rows if row["sector"] == sector], ddof=1)),
                }
                for metric in ("image_drop", "patient_drop")
            }
            for sector in SECTOR_KEYS
        },
        "pairwise_mann_whitney_image": image_pairs,
        "pairwise_mann_whitney_patient": patient_pairs,
        "csv": os.path.relpath(csv_path, RESULTS),
    }
    json_path = os.path.join(METRICS_DIR, f"vvit_permutation_importance_{args.output_tag}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    importance_path, roc_path = save_figures(
        rows, labels, baseline_scores, patient_y, patient_scores, args.output_tag, figure_dir,
    )
    print(json.dumps(summary["importance"], indent=2), flush=True)
    print(f"Saved {json_path}, {importance_path}, {roc_path}", flush=True)


if __name__ == "__main__":
    main()
