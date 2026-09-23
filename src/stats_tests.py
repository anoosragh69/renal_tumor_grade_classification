"""
DeLong test for comparing two AUCs (plan §4 Step 9.3 / reference §7)
====================================================================
U-statistic implementation (no standard library ships DeLong). Based on the
well-known fast-DeLong formulation (Sun & Xu, fast DeLong):

  V01, V02 = placement values of positive scores under each model
  V11, V12 = placement values of negative scores
  AUC_i = mean(V0i) over positives  (equivalently 1 - mean(V1i))
  covariance via sample-based variance of the placement values

Returns z, p-value for H0: AUC_a == AUC_b.

Usage:
    python src/stats_tests.py              # self-test on synthetic predictions
    from src.stats_tests import delong_test
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats as sps


def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks (average ranks, 1-based) for possibly tied values."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        # ranks i+1 .. j+1 (1-based), average
        ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(n, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def _auc_values(scores: np.ndarray, labels: np.ndarray):
    """
    Returns (auc, V0, V1) where
      V0 = placement values for positives  (n_pos,)
      V1 = placement values for negatives  (n_neg,)
    auc = mean(V0)
    """
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels).astype(int).ravel()
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("need both classes for AUC")

    z = np.concatenate([scores, scores])
    z0 = np.concatenate([np.full(n_pos, -np.inf), neg])   # negatives as refs for V0?
    # Standard fast DeLong construction:
    # h = concat(pos_scores, neg_scores) as "predictions" being compared
    # Actually use the canonical form from fastdelong:
    #   xy = outer-based placement
    all_s = np.concatenate([pos, neg])
    # For each positive score, fraction of negatives it beats + 0.5 ties
    # vectorized via midrank on concatenated pairs is the classic approach.

    # Classic: midrank all scores; then
    # V0[i] = (midrank(pos_i) among concat(pos, neg_i-specific)...)
    # Use the clean O(n^2) definition (dataset is small):
    # V0: for each positive, mean over negatives of (s_pos > s_neg) + 0.5*
    #     (s_pos == s_neg)
    cmp_pos = pos[:, None] - neg[None, :]           # (n_pos, n_neg)
    V0 = ((cmp_pos > 0).sum(axis=1) + 0.5 * (cmp_pos == 0).sum(axis=1)) / n_neg
    # V1: for each negative, mean over positives of (s_pos > s_neg) + 0.5 ties
    V1 = ((cmp_pos > 0).sum(axis=0) + 0.5 * (cmp_pos == 0).sum(axis=0)) / n_pos
    auc = float(V0.mean())
    return auc, V0, V1


def delong_roc_test(y_true, score_a, score_b):
    """
    Two-sided DeLong test comparing AUC(score_a) vs AUC(score_b) on the same
    labels (paired ROC curves).

    Returns dict(auc_a, auc_b, diff, var, z, pvalue)
    """
    y = np.asarray(y_true).astype(int).ravel()
    s_a = np.asarray(score_a, dtype=float).ravel()
    s_b = np.asarray(score_b, dtype=float).ravel()
    if not (len(y) == len(s_a) == len(s_b)):
        raise ValueError("length mismatch")

    auc_a, V0a, V1a = _auc_values(s_a, y)
    auc_b, V0b, V1b = _auc_values(s_b, y)
    n_pos = len(V0a)
    n_neg = len(V1a)

    # Covariance of AUCs via placement-value sample covariance (fast DeLong)
    # S01 = cov of V0a, V0b across positives; S11 across negatives
    def _cov(x, y_):
        n = len(x)
        if n <= 1:
            return 0.0
        return float(np.cov(x, y_, ddof=1)[0, 1])

    # Var(AUC_a) = cov(V0a,V0a)/n_pos + cov(V1a,V1a)/n_neg
    # Cov(AUC_a, AUC_b) = cov(V0a,V0b)/n_pos + cov(V1a,V1b)/n_neg
    S0_aa = _cov(V0a, V0a)
    S0_bb = _cov(V0b, V0b)
    S0_ab = _cov(V0a, V0b)
    S1_aa = _cov(V1a, V1a)
    S1_bb = _cov(V1b, V1b)
    S1_ab = _cov(V1a, V1b)

    var_a = S0_aa / n_pos + S1_aa / n_neg
    var_b = S0_bb / n_pos + S1_bb / n_neg
    cov_ab = S0_ab / n_pos + S1_ab / n_neg

    diff = auc_a - auc_b
    var = var_a + var_b - 2 * cov_ab
    if var <= 0:
        # numerical edge: identical scores or degenerate
        if abs(diff) < 1e-12:
            z, p = 0.0, 1.0
        else:
            z = float(np.sign(diff) * np.inf)
            p = 0.0
    else:
        z = float(diff / np.sqrt(var))
        p = float(2 * sps.norm.sf(abs(z)))

    return {
        "auc_a": auc_a, "auc_b": auc_b, "diff": float(diff),
        "var": float(var), "z": z, "pvalue": p,
        "var_a": float(var_a), "var_b": float(var_b), "cov_ab": float(cov_ab),
    }


# alias
delong_test = delong_roc_test


def _self_test():
    print("=== DeLong self-test ===")
    rng = np.random.default_rng(42)
    n_pos, n_neg = 40, 60
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

    # Model A: well-separated; Model B: noisier
    scores_a = np.concatenate([
        rng.normal(1.0, 1.0, n_pos),
        rng.normal(0.0, 1.0, n_neg),
    ])
    scores_b = np.concatenate([
        rng.normal(0.3, 1.5, n_pos),
        rng.normal(0.0, 1.5, n_neg),
    ])

    # Cross-check AUC against evaluation.auroc / sklearn
    from src.training.evaluation import auroc
    r = delong_roc_test(y, scores_a, scores_b)
    auc_eval = auroc(y, scores_a)
    print(f"DeLong AUC_a={r['auc_a']:.4f} (eval.auroc={auc_eval:.4f}) "
          f"AUC_b={r['auc_b']:.4f} diff={r['diff']:.4f} z={r['z']:.3f} p={r['pvalue']:.4f}")
    assert abs(r["auc_a"] - auc_eval) < 1e-9, (r["auc_a"], auc_eval)

    try:
        from sklearn.metrics import roc_auc_score
        sk = roc_auc_score(y, scores_a)
        assert abs(sk - r["auc_a"]) < 1e-9
        print(f"sklearn AUC match: {sk:.6f}")
    except ImportError:
        pass

    # Self-comparison: A vs A -> diff=0, p=1
    r0 = delong_roc_test(y, scores_a, scores_a)
    assert abs(r0["diff"]) < 1e-12, r0
    assert abs(r0["pvalue"] - 1.0) < 1e-9, r0
    print(f"self-comparison: diff={r0['diff']:.2e} p={r0['pvalue']:.6f}")

    # Perfect vs random
    perfect = y.astype(float) + rng.uniform(0, 0.01, len(y))
    noise = rng.uniform(0, 1, len(y))
    r1 = delong_roc_test(y, perfect, noise)
    assert r1["auc_a"] > 0.99 and r1["pvalue"] < 0.05, r1
    print(f"perfect vs noise: AUC {r1['auc_a']:.3f} vs {r1['auc_b']:.3f} p={r1['pvalue']:.2e}")

    # Identical AUCs but independent noise — p should often be non-sig;
    # just check validity
    assert 0 <= r["pvalue"] <= 1

    print("\nstats_tests.py self-test PASSED.")


if __name__ == "__main__":
    _self_test()
