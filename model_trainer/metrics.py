"""Evaluation metrics for financial ML models."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TOPK_VALUES = [1, 3, 5, 10]


def compute_prior_abtype_frequencies(
    df: pd.DataFrame,
    y_col: str,
    group_cols: Optional[List[str]] = None,
) -> pd.Series:
    """Compute each abtype's frequency of being the top-1 (highest y) in a query.

    Args:
        df: DataFrame with abtype column.
        y_col: Label column name.
        group_cols: Columns to group by.

    Returns:
        pd.Series indexed by abtype, values are proportions (sum=1).
        Empty Series if abtype column is missing.
    """
    if group_cols is None:
        group_cols = ["date", "time_sec", "code"]

    if "abtype" not in df.columns:
        return pd.Series(dtype=float)

    top1_abtypes = []
    for _, group in df.groupby(group_cols, sort=False):
        if len(group) > 0:
            top1_idx = group[y_col].idxmax()
            top1_abtypes.append(group.loc[top1_idx, "abtype"])

    if not top1_abtypes:
        return pd.Series(dtype=float)

    return pd.Series(top1_abtypes).value_counts(normalize=True)


def build_prior_topk_abtypes(
    df: pd.DataFrame,
    y_col: str,
    group_cols: Optional[List[str]] = None,
    topk_values: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """Build prior topk abtypes mapping from train+val data.

    Args:
        df: DataFrame (typically train + val concatenated).
        y_col: Label column name.
        group_cols: Columns to group by.
        topk_values: List of k values.

    Returns:
        Dict mapping k -> np.ndarray of abtype values that are the top-k
        most frequent abtypes for being top-1 in a query.
    """
    if topk_values is None:
        topk_values = TOPK_VALUES

    freq = compute_prior_abtype_frequencies(df, y_col, group_cols)
    prior: Dict[int, np.ndarray] = {}
    for k in topk_values:
        actual_k = min(k, len(freq))
        if actual_k > 0:
            prior[k] = freq.head(actual_k).index.values.astype(np.int64)
        else:
            prior[k] = np.array([], dtype=np.int64)
    return prior


def evaluate(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    split_name: str = "",
    df: Optional[pd.DataFrame] = None,
    topk: int = 1,
    prior_topk_abtypes: Optional[Dict[int, np.ndarray]] = None,
) -> dict:
    """Compute metrics for one data split.

    Args:
        y_pred:     Model predictions (1-D).
        y_true:     True labels (1-D).
        split_name: Label used in printed output (e.g. "val", "test").
        df:         DataFrame with date, time_sec, code columns (for grouped metrics).
        topk:       Number of top items to consider for hit rate calculation.
        prior_topk_abtypes: Dict mapping k -> np.ndarray of prior topk abtype values,
                            used for topk_vs_prior metrics.

    Returns:
        dict with keys: n_samples, topk_hit_rate, excess_hit_rate, avg_pred_rank,
                        top{k}_above_mean_rate for k in TOPK_VALUES,
                        top{k}_vs_prior_rate for k in TOPK_VALUES (if prior provided).
    """
    result: dict = {"n_samples": int(len(y_pred))}

    tag = f"[{split_name}] " if split_name else ""
    log_msg = f"  {tag}n_samples={result['n_samples']}"

    if df is not None and len(df) == len(y_pred):
        required_cols = ["date", "time_sec", "code"]
        if all(col in df.columns for col in required_cols):
            group_hit_rates = []
            group_excess_rates = []
            group_pred_ranks = []

            topk_above_mean: Dict[int, list] = {k: [] for k in TOPK_VALUES}

            has_prior = (
                prior_topk_abtypes is not None
                and "abtype" in df.columns
            )
            topk_vs_prior: Optional[Dict[int, list]] = (
                {k: [] for k in TOPK_VALUES} if has_prior else None
            )
            abtype_values = df["abtype"].values if has_prior else None

            grouped = df.groupby(required_cols, sort=False)

            for group_idx in grouped.indices.values():
                group_pred = y_pred[group_idx]
                group_true = y_true[group_idx]
                size = len(group_idx)
                group_mean = np.mean(group_true)

                k = min(topk, size)
                if k > 0:
                    true_topk_indices = np.argsort(group_true)[-k:]
                    pred_topk_indices = np.argsort(group_pred)[-k:]

                    hits = len(np.intersect1d(true_topk_indices, pred_topk_indices))
                    group_hit_rate = hits / k
                    group_hit_rates.append(group_hit_rate)

                    baseline = k / size
                    excess_rate = group_hit_rate - baseline
                    group_excess_rates.append(excess_rate)

                    pred_top1_idx = pred_topk_indices[-1]
                    true_rank = np.argsort(np.argsort(group_true))[pred_top1_idx]
                    pred_rank = true_rank / (size - 1) if size > 1 else 0.0
                    group_pred_ranks.append(pred_rank)

                for topk_val in TOPK_VALUES:
                    actual_k = min(topk_val, size)
                    if actual_k > 0:
                        pred_topk_idx = np.argsort(group_pred)[-actual_k:]
                        pred_topk_mean = np.mean(group_true[pred_topk_idx])
                        topk_above_mean[topk_val].append(
                            1.0 if pred_topk_mean >= group_mean else 0.0
                        )

                if has_prior and abtype_values is not None:
                    group_abtypes = abtype_values[group_idx]
                    for topk_val in TOPK_VALUES:
                        actual_k = min(topk_val, size)
                        prior_abtypes_k = prior_topk_abtypes.get(
                            topk_val, np.array([])
                        )
                        if actual_k > 0 and len(prior_abtypes_k) > 0:
                            pred_topk_idx = np.argsort(group_pred)[-actual_k:]
                            pred_topk_mean = np.mean(group_true[pred_topk_idx])
                            prior_mask = np.isin(group_abtypes, prior_abtypes_k)
                            if prior_mask.any():
                                prior_mean = np.mean(group_true[prior_mask])
                                topk_vs_prior[topk_val].append(
                                    1.0 if pred_topk_mean >= prior_mean else 0.0
                                )

            if group_hit_rates:
                result["topk_hit_rate"] = float(np.mean(group_hit_rates))
                result["excess_hit_rate"] = float(np.mean(group_excess_rates))
                result["avg_pred_rank"] = float(np.mean(group_pred_ranks))

                log_msg += (
                    f" | Top-{topk} Hit Rate={result['topk_hit_rate']:.4f} "
                    f"| Excess Hit Rate={result['excess_hit_rate']:.4f} "
                    f"| Avg Pred Rank={result['avg_pred_rank']:.4f}"
                )

            for topk_val in TOPK_VALUES:
                vals = topk_above_mean[topk_val]
                if vals:
                    key = f"top{topk_val}_above_mean_rate"
                    result[key] = float(np.mean(vals))
                    log_msg += f" | Top{topk_val} Above Mean={result[key]:.4f}"

            if has_prior and topk_vs_prior is not None:
                for topk_val in TOPK_VALUES:
                    vals = topk_vs_prior[topk_val]
                    if vals:
                        key = f"top{topk_val}_vs_prior_rate"
                        result[key] = float(np.mean(vals))
                        log_msg += f" | Top{topk_val} vs Prior={result[key]:.4f}"

    print(log_msg)
    return result
