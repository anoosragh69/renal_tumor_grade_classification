"""
Task 13: Evaluation Metrics
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import torch
from collections import Counter


def compute_accuracy(y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true) if len(y_true) > 0 else 0


def compute_f1(y_true, y_pred, num_classes=4, average="weighted"):
    f1_per_class = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_per_class.append(f1)

    if average == "macro":
        return np.mean(f1_per_class), f1_per_class
    elif average == "weighted":
        counts = Counter(y_true)
        weights = [counts.get(c, 0) for c in range(num_classes)]
        total = sum(weights)
        weighted_f1 = sum(f * w for f, w in zip(f1_per_class, weights)) / total if total > 0 else 0
        return weighted_f1, f1_per_class
    return f1_per_class


def confusion_matrix(y_true, y_pred, num_classes=4):
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def confusion_matrix_str(cm, class_names=None):
    if class_names is None:
        class_names = [f"Grade {i+1}" for i in range(len(cm))]

    max_name = max(len(n) for n in class_names)
    header = " " * (max_name + 2) + "  ".join(f"{n:>8}" for n in class_names)
    lines = [header, "-" * len(header)]

    for i, row in enumerate(cm):
        line = f"{class_names[i]:<{max_name}}  " + "  ".join(f"{v:8d}" for v in row)
        lines.append(line)

    return "\n".join(lines)


def per_class_metrics(y_true, y_pred, num_classes=4, class_names=None):
    if class_names is None:
        class_names = [f"Grade {i+1}" for i in range(num_classes)]

    results = []
    for c in range(num_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != c and p != c)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        support = sum(1 for t in y_true if t == c)

        results.append({
            "class": class_names[c],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "support": support,
        })
    return results


def evaluate(y_true, y_pred, num_classes=4, class_names=None):
    if class_names is None:
        class_names = [f"Grade {i+1}" for i in range(num_classes)]

    acc = compute_accuracy(y_true, y_pred)
    weighted_f1, f1_per_class = compute_f1(y_true, y_pred, num_classes, "weighted")
    macro_f1, _ = compute_f1(y_true, y_pred, num_classes, "macro")
    cm = confusion_matrix(y_true, y_pred, num_classes)
    pcm = per_class_metrics(y_true, y_pred, num_classes, class_names)

    report = {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class": pcm,
    }

    print(f"Accuracy: {acc:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(confusion_matrix_str(cm, class_names))
    print(f"\nPer-Class Metrics:")
    for m in pcm:
        print(f"  {m['class']}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} Spec={m['specificity']:.3f} Support={m['support']}")

    return report


if __name__ == "__main__":
    y_true = [0, 1, 1, 2, 2, 2, 3, 3, 1, 0]
    y_pred = [0, 1, 2, 2, 1, 2, 3, 1, 1, 0]

    print("=== Evaluation Metrics Test ===\n")
    report = evaluate(y_true, y_pred, num_classes=4)
    print("\nMetrics test passed!")
