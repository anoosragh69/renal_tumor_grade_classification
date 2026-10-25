"""Run the saved vViT on one patient from the checkpoint's held-out test split."""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data.dataset import RenalTumorSectorDataset, get_paths
from src.models.fusion_baseline_vote import majority_vote
from src.models.vvit import TOKEN_ORDER, VViT
from src.training.evaluation import aggregate_patient
from src.training.run_training import CKPT_DIR


DEFAULT_CHECKPOINT = os.path.join(CKPT_DIR, "vvit_paper_seed42_best.pt")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--patient-id", help="Test-split patient; defaults to the first sorted ID")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    preprocessing = checkpoint.get("preprocessing")
    if not preprocessing or "radiomic_mean" not in preprocessing:
        raise ValueError("Checkpoint lacks the train-only scaler metadata required for inference")
    paths = get_paths(
        splits_file=checkpoint["data_paths"]["splits"],
        top16_file=checkpoint["data_paths"]["top16"],
        radiomics_file=checkpoint["data_paths"]["radiomics"],
    )
    dataset = RenalTumorSectorDataset(
        split="test", augment=False, paths=paths, preprocessing=preprocessing,
    )
    patient_ids = sorted({pid for pid, _ in dataset.index})
    if not patient_ids:
        raise ValueError("The checkpoint's test split has no available crop/radiomics rows")
    patient_id = args.patient_id or patient_ids[0]
    if patient_id not in dataset.patient_ids:
        raise ValueError(f"{patient_id} is not in the checkpoint's held-out test split")

    indices = [i for i, (pid, _) in enumerate(dataset.index) if pid == patient_id]
    if not indices:
        raise ValueError(f"No processed test slices are available for {patient_id}")
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)
    model = VViT(**checkpoint.get("model_config", {})).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    slice_rows = []
    labels = []
    with torch.inference_mode():
        for batch in loader:
            batch_device = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
                if key in TOKEN_ORDER[1:]
            }
            logits = model(batch_device).float()
            probs = torch.sigmoid(logits)
            votes = (probs >= 0.5).long()
            final_votes = majority_vote(logits)
            mean_probs = probs.mean(dim=1)
            slice_indices = batch["slice_idx"].numpy().astype(int)
            batch_labels = batch["label"].numpy().astype(int)
            for i, slice_idx in enumerate(slice_indices):
                slice_rows.append({
                    "slice_idx": int(slice_idx),
                    "votes": votes[i].cpu().numpy().astype(int),
                    "probabilities": probs[i].cpu().numpy(),
                    "final_vote": int(final_votes[i].item()),
                    "mean_probability": float(mean_probs[i].item()),
                })
            labels.extend(batch_labels.tolist())

    slice_rows.sort(key=lambda row: row["slice_idx"])
    labels = np.asarray(labels, dtype=int)
    slice_preds = np.asarray([row["final_vote"] for row in slice_rows], dtype=int)
    slice_scores = np.asarray([row["mean_probability"] for row in slice_rows])
    patient_vote = aggregate_patient(slice_preds, slice_scores, [patient_id] * len(slice_rows), mode="vote")
    patient_mean = aggregate_patient(slice_preds, slice_scores, [patient_id] * len(slice_rows), mode="mean")
    patient_pred = patient_vote[patient_id][0]
    patient_probability = patient_mean[patient_id][1]
    true_label = int(labels[0])
    if not np.all(labels == true_label):
        raise ValueError(f"Inconsistent labels across {patient_id} slices")

    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch')}) on {device}")
    print(f"Patient: {patient_id} | observed label: {'high' if true_label else 'low'} grade")
    print("Input requirement: CT crop + tumor mask-derived radiomics + kits.json clinical fields.")
    print("Per-slice head outputs are attended model heads, not independent sector classifiers.")
    print("slice\t" + "\t".join(f"{name}_vote" for name in TOKEN_ORDER)
          + "\tfinal_vote\tP(high)_mean_heads")
    for row in slice_rows:
        head_votes = "\t".join(str(int(v)) for v in row["votes"])
        print(f"{row['slice_idx']}\t{head_votes}\t{row['final_vote']}\t"
              f"{row['mean_probability']:.4f}")
    print(f"Patient result: {'high' if patient_pred else 'low'} grade; "
          f"P(high)={patient_probability:.4f}; {len(slice_rows)} slices")
    print("Research illustration only; not for diagnosis.")


if __name__ == "__main__":
    main()
