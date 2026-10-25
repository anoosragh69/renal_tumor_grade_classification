"""Compare vViT and image-only baselines on their shared patient test set."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd

from src.stats_tests import delong_roc_test
from src.training.evaluation import aggregate_patient, mcnemar_test
from src.training.run_training import METRICS_DIR, RESULTS


def read_patient_predictions(path, model):
    frame = pd.read_csv(path)
    required = {"patient_id", "label"}
    if model == "vvit":
        required |= {"vote_pred", "mean_prob"}
        pred_col, score_col = "vote_pred", "mean_prob"
    else:
        required |= {"pred", "prob"}
        pred_col, score_col = "pred", "prob"
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path} contains missing prediction values")

    labels_by_patient = frame.groupby("patient_id")["label"].nunique()
    if (labels_by_patient != 1).any():
        raise ValueError(f"{path} has inconsistent slice labels within a patient")
    patient_pred = aggregate_patient(
        frame[pred_col].to_numpy(), frame[score_col].to_numpy(),
        frame["patient_id"].to_numpy(), mode="vote",
    )
    patient_score = aggregate_patient(
        frame[pred_col].to_numpy(), frame[score_col].to_numpy(),
        frame["patient_id"].to_numpy(), mode="mean",
    )
    pids = sorted(patient_pred)
    return {
        "patient_ids": pids,
        "labels": np.array([
            int(frame.loc[frame.patient_id == pid, "label"].iloc[0]) for pid in pids
        ]),
        "predictions": np.array([patient_pred[pid][0] for pid in pids], dtype=int),
        "scores": np.array([patient_score[pid][1] for pid in pids], dtype=float),
    }


def compare(reference, candidate, candidate_name):
    if reference["patient_ids"] != candidate["patient_ids"]:
        raise ValueError(f"{candidate_name} does not use the same patient IDs as vViT")
    if not np.array_equal(reference["labels"], candidate["labels"]):
        raise ValueError(f"{candidate_name} labels differ from vViT labels")
    y = reference["labels"]
    delong = delong_roc_test(y, reference["scores"], candidate["scores"])
    mcnemar = mcnemar_test(y, reference["predictions"], candidate["predictions"], exact=True)
    return {
        "model": candidate_name,
        "n_patients": len(y),
        "vvit_accuracy": float(np.mean(reference["predictions"] == y)),
        "baseline_accuracy": float(np.mean(candidate["predictions"] == y)),
        "vvit_auroc": delong["auc_a"],
        "baseline_auroc": delong["auc_b"],
        "auroc_difference_vvit_minus_baseline": delong["diff"],
        "delong_z": delong["z"],
        "delong_pvalue": delong["pvalue"],
        "mcnemar_vvit_correct_baseline_wrong": mcnemar["b"],
        "mcnemar_vvit_wrong_baseline_correct": mcnemar["c"],
        "mcnemar_statistic": mcnemar["statistic"],
        "mcnemar_pvalue": mcnemar["pvalue"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default=METRICS_DIR)
    parser.add_argument("--vvit-file", default="test_predictions_paper_seed42.csv")
    parser.add_argument("--output-tag", default="seed42")
    args = parser.parse_args(argv)

    root = args.metrics_dir
    reference = read_patient_predictions(os.path.join(root, args.vvit_file), "vvit")
    comparisons = []
    for model in ("vit", "convnext", "resnext"):
        candidate = read_patient_predictions(
            os.path.join(root, f"baseline_{model}_test_predictions.csv"), model
        )
        comparisons.append(compare(reference, candidate, model))

    summary = {
        "reference": "vViT paper_seed42",
        "aggregation": "patient hard labels by majority vote; patient AUROC scores by mean slice probability",
        "test_patients": len(reference["patient_ids"]),
        "positive_patients": int(reference["labels"].sum()),
        "comparisons": comparisons,
        "interpretation": (
            "Exploratory paired comparisons on one shared test set (n=31); low power, "
            "with no multiplicity adjustment across three baseline comparisons."
        ),
    }
    output_json = os.path.join(root, f"vvit_baseline_comparisons_{args.output_tag}.json")
    output_csv = os.path.join(root, f"vvit_baseline_comparisons_{args.output_tag}.csv")
    os.makedirs(root, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(comparisons).to_csv(output_csv, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Saved {output_json} and {output_csv}")


if __name__ == "__main__":
    main()
