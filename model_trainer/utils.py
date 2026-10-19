"""Utility functions for segment operations and ranking metrics."""
from __future__ import annotations

from typing import List

import torch


def segment_softmax(
        logits: torch.Tensor,
        group_indices: torch.Tensor,
        num_groups: int,
) -> torch.Tensor:
    """Compute softmax independently within each group using scatter operations.

    Numerically stable: subtracts per-group max before exponentiation.

    Args:
        logits: Flat tensor of raw logits for all samples, shape (N,).
        group_indices: Group membership index for each sample, shape (N,).
        num_groups: Total number of distinct groups.

    Returns:
        Softmax probabilities for each sample within its group, shape (N,).
    """
    group_max_logits = torch.zeros(num_groups, device=logits.device)
    group_max_logits.scatter_reduce_(0, group_indices, logits, reduce='amax', include_self=False)

    shifted_logits = logits - group_max_logits[group_indices]
    exp_logits = torch.exp(shifted_logits)

    group_sum_exp = torch.zeros(num_groups, device=logits.device)
    group_sum_exp.scatter_add_(0, group_indices, exp_logits)

    return exp_logits / group_sum_exp[group_indices]


def segment_sum(
        values: torch.Tensor,
        group_indices: torch.Tensor,
        num_groups: int,
) -> torch.Tensor:
    """Compute sum independently within each group using scatter_add.

    Args:
        values: Flat tensor of values for all samples, shape (N,).
        group_indices: Group membership index for each sample, shape (N,).
        num_groups: Total number of distinct groups.

    Returns:
        Per-group sums, shape (num_groups,).
    """
    group_sums = torch.zeros(num_groups, device=values.device)
    group_sums.scatter_add_(0, group_indices, values)
    return group_sums


def compute_topk_hit_rate(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        group_sizes: List[int],
        topk: int = 1,
) -> float:
    """Compute average top-k hit rate across all groups using vectorized operations.

    For each group, counts the intersection between predicted top-k and true top-k,
    then divides by k. Returns the mean across all groups.

    Args:
        predictions: Flat tensor of predicted scores for all samples, shape (N,).
        targets: Flat tensor of true target values for all samples, shape (N,).
        group_sizes: List of sample counts per group.
        topk: Number of top items to consider for hit rate.

    Returns:
        Mean top-k hit rate across all groups, or 0.0 if no groups.
    """
    if not group_sizes:
        return 0.0

    pred_groups = torch.split(predictions, group_sizes)
    target_groups = torch.split(targets, group_sizes)

    total_hits = 0
    total_topk = 0

    for group_pred, group_target, group_size in zip(pred_groups, target_groups, group_sizes):
        effective_k = min(topk, group_size)

        _, true_topk_indices = torch.topk(group_target, effective_k)
        _, pred_topk_indices = torch.topk(group_pred, effective_k)

        true_mask = torch.zeros(group_size, dtype=torch.bool, device=predictions.device)
        pred_mask = torch.zeros(group_size, dtype=torch.bool, device=predictions.device)
        true_mask.scatter_(0, true_topk_indices, True)
        pred_mask.scatter_(0, pred_topk_indices, True)

        intersection_count = (true_mask & pred_mask).sum().item()
        total_hits += intersection_count
        total_topk += effective_k

    return total_hits / total_topk if total_topk > 0 else 0.0
