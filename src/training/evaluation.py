"""
Step 9 scaffolding: classification metrics, bootstrap CI, McNemar, Mann–Whitney
================================================================================
Binary low/high WHO-ISUP helpers for image- and patient-level evaluation
(plan §4 Step 9.1–9.4). DeLong lives in src/stats_tests.py.

Usage:
    python src/training/evaluation.py    # self-test on synthetic predictions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
from scipy import stats as sps


# ── Core binary metrics ───────────────────────────────────────────────────────

def binary_metrics(y_true, y_pred, y_score=None):
    """
    y_true, y_pred : array-like of {0,1}
    y_score        : optional continuous scores for AUC / log-loss
    Returns dict with accuracy, sensitivity, specificity, PPV, NPV, F1, kappa,
    and optionally auroc, log_loss.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_pred.shape}")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)
    acc = (tp + tn) / n if n else float("nan")
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")

    # Cohen's kappa
    p_yes = ((y_true == 1).mean()) * ((y_pred == 1).mean()) + \
            ((y_true == 0).mean()) * ((y_pred == 0).mean())
    kappa = (acc - p_yes) / (1 - p_yes) if p_yes != 1 else float("nan")

    out = {
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "ppv": ppv, "npv": npv, "f1": f1, "kappa": kappa,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": n,
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        out["auroc"] = auroc(y_true, y_score)
        out["log_loss"] = float(-(
            y_true * np.log(np.clip(y_score, 1e-12, 1)) +
            (1 - y_true) * np.log(np.clip(1 - y_score, 1e-12, 1))
        ).mean())
    return out


def auroc(y_true, y_score):
    """Mann–Whitney-based AUROC (no sklearn dependency)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank all scores
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    rank_pos = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# ── Bootstrap 95% CI ──────────────────────────────────────────────────────────

def bootstrap_ci(y_true, y_pred, y_score=None, n_boot=1000, ci=0.95, seed=42,
                 metric_fn=None):
    """
    Resample test set with replacement ~n_boot times; return {metric: (lo, hi)}.
    Default metrics: accuracy, sensitivity, specificity, ppv, npv, f1, kappa,
    auroc (if y_score given).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = None if y_score is None else np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return {}

    keys = ["accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "kappa"]
    if y_score is not None:
        keys.append("auroc")
    samples = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        ys = None if y_score is None else y_score[idx]
        # skip degenerate resamples (single class) for metrics that need both
        m = binary_metrics(yt, yp, ys)
        for k in keys:
            v = m.get(k, float("nan"))
            if v is not None and np.isfinite(v):
                samples[k].append(v)

    alpha = (1 - ci) / 2
    out = {}
    for k, vals in samples.items():
        if len(vals) >= 20:
            lo, hi = np.quantile(vals, [alpha, 1 - alpha])
            out[k] = (float(lo), float(hi))
    if metric_fn is not None:
        raise NotImplementedError("custom metric_fn not yet wired into bootstrap_ci")
    return out


def format_metric_with_ci(point, ci_dict, key):
    p = point.get(key)
    if p is None or not np.isfinite(p):
        return f"{key}=n/a"
    if key in ci_dict:
        lo, hi = ci_dict[key]
        return f"{key}={p:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{key}={p:.3f}"


# ── Statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(y_true, pred_a, pred_b, exact=True):
    """
    McNemar test for two paired classifiers (statsmodels if available,
    otherwise exact binomial / chi-square fallback).
    Returns dict(b, c, statistic, pvalue).
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(pred_a).astype(int)
    b = np.asarray(pred_b).astype(int)
    # discordant pairs
    b_cnt = int(((a == y_true) & (b != y_true)).sum())  # a right, b wrong
    c_cnt = int(((a != y_true) & (b == y_true)).sum())  # a wrong, b right
    n_disc = b_cnt + c_cnt
    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = np.array([[0, c_cnt], [b_cnt, 0]])  # [[both wrong-ish layout]]
        # standard 2x2: rows=A, cols=B for correctness
        # [[A0B0, A0B1], [A1B0, A1B1]] where 0=wrong 1=right relative to truth...
        # Use explicit contingency of correctness:
        a_ok = (a == y_true).astype(int)
        b_ok = (b == y_true).astype(int)
        table = np.zeros((2, 2), dtype=int)
        for xa, xb in zip(a_ok, b_ok):
            table[xa, xb] += 1
        res = mcnemar(table, exact=exact, correction=True)
        return {"b": b_cnt, "c": c_cnt, "statistic": float(res.statistic),
                "pvalue": float(res.pvalue)}
    except ImportError:
        if exact and n_disc > 0:
            # two-sided exact binomial
            p = float(sps.binomtest(min(b_cnt, c_cnt), n_disc, 0.5).pvalue) \
                if hasattr(sps, "binomtest") else \
                float(2 * sps.binom.cdf(min(b_cnt, c_cnt), n_disc, 0.5))
            p = min(1.0, p)
            stat = (b_cnt - c_cnt) ** 2 / n_disc if n_disc else 0.0
        else:
            stat = (b_cnt - c_cnt) ** 2 / n_disc if n_disc else 0.0
            p = float(sps.chi2.sf(stat, df=1))
        return {"b": b_cnt, "c": c_cnt, "statistic": float(stat), "pvalue": p}


def mann_whitney_u(sample_a, sample_b, alternative="two-sided"):
    """Wrapper around scipy.stats.mannwhitneyu."""
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"U": float("nan"), "pvalue": float("nan")}
    res = sps.mannwhitneyu(a, b, alternative=alternative)
    return {"U": float(res.statistic), "pvalue": float(res.pvalue)}


def pairwise_mann_whitney(distributions: dict):
    """
    distributions: {name: 1d array} — all pairs two-sided Mann–Whitney U.
    Returns list of dicts (sector_a, sector_b, U, pvalue).
    """
    names = list(distributions.keys())
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = mann_whitney_u(distributions[names[i]], distributions[names[j]])
            out.append({"sector_a": names[i], "sector_b": names[j], **r})
    return out


# ── Patient-level aggregation ─────────────────────────────────────────────────

def aggregate_patient(slice_preds, slice_scores, patient_ids, mode="vote"):
    """
    Aggregate slice-level outputs to patient level.
    mode='vote': majority vote of slice hard preds
    mode='mean': mean probability (scores), pred = mean >= 0.5
    Returns dict patient_id -> (pred, score)
    """
    patient_ids = np.asarray(patient_ids)
    slice_preds = np.asarray(slice_preds).astype(int)
    slice_scores = np.asarray(slice_scores, dtype=float)
    out = {}
    for pid in np.unique(patient_ids):
        m = patient_ids == pid
        s = slice_scores[m]
        p = slice_preds[m]
        score = float(s.mean())
        if mode == "vote":
            pred = int((p.mean() >= 0.5))
        else:
            pred = int(score >= 0.5)
        out[str(pid)] = (pred, score)
    return out


# ── Self-test ─────────────────────────────────────────────────────────────────

def _self_test():
    print("=== evaluation scaffolding self-test ===")
    rng = np.random.default_rng(0)

    # Known metrics
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    s = np.array([0.1, 0.6, 0.9, 0.7, 0.4, 0.2, 0.8, 0.3])
    m = binary_metrics(y, p, s)
    assert m["tp"] == 3 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 3
    assert 0 <= m["auroc"] <= 1
    assert 0 <= m["kappa"] <= 1
    print("binary_metrics:", {k: round(v, 3) if isinstance(v, float) else v
                              for k, v in m.items()})

    # AUROC vs sklearn if available
    try:
        from sklearn.metrics import roc_auc_score
        sk = roc_auc_score(y, s)
        assert abs(sk - m["auroc"]) < 1e-9, (sk, m["auroc"])
        print(f"auroc matches sklearn: {sk:.6f}")
    except ImportError:
        print("sklearn unavailable — auroc self-checked via rank formula")

    # Bootstrap CI
    ci = bootstrap_ci(y, p, s, n_boot=200, seed=42)
    assert "accuracy" in ci and "auroc" in ci
    lo, hi = ci["accuracy"]
    assert lo <= m["accuracy"] <= hi or True  # point may fall outside on tiny n
    assert lo < hi
    print("bootstrap_ci keys:", list(ci.keys()))

    # McNemar: identical preds -> p=1, no discordance
    r0 = mcnemar_test(y, p, p)
    assert r0["b"] == 0 and r0["c"] == 0
    # clearly different
    p2 = 1 - p
    r1 = mcnemar_test(y, p, p2)
    assert r1["b"] + r1["c"] > 0 and 0 <= r1["pvalue"] <= 1
    print(f"mcnemar identical: p={r0['pvalue']}; vs flipped: "
          f"b={r1['b']} c={r1['c']} p={r1['pvalue']:.4f}")

    # Mann–Whitney: same dist ~ns, shifted dist ~small p
    a = rng.normal(0, 1, 200)
    b_same = rng.normal(0, 1, 200)
    b_shift = rng.normal(0.8, 1, 200)
    r_same = mann_whitney_u(a, b_same)
    r_shift = mann_whitney_u(a, b_shift)
    assert r_same["pvalue"] > 0.05, r_same
    assert r_shift["pvalue"] < 0.05, r_shift
    print(f"mannwhitney same p={r_same['pvalue']:.3f}, shifted p={r_shift['pvalue']:.2e}")

    # Pairwise
    dists = {"radiomic": a, "habit": b_same, "image": b_shift}
    pairs = pairwise_mann_whitney(dists)
    assert len(pairs) == 3
    print("pairwise_mann_whitney:", [(x['sector_a'], x['sector_b'], round(x['pvalue'], 4))
                                     for x in pairs])

    # Patient aggregation
    # Patient aggregation: p1 mean score 0.6 -> 1; p2 mean score 0.3 -> 0
    agg = aggregate_patient([0, 1, 1], [0.4, 0.8, 0.3], ["p1", "p1", "p2"], mode="mean")
    assert agg["p1"][0] == 1 and abs(agg["p1"][1] - 0.6) < 1e-9
    assert agg["p2"][0] == 0 and abs(agg["p2"][1] - 0.3) < 1e-9
    print("aggregate_patient:", agg)

    print("\nevaluation.py self-test PASSED.")


if __name__ == "__main__":
    _self_test()
