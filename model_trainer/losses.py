"""Custom loss functions for training neural ranking models.

Supports:
    - PearsonLoss: Differentiable Pearson correlation loss (1 - pearson_r).
    - ListNetLoss: Cross-entropy between predicted and target probability distributions.
    - ListMLELoss: Negative log-likelihood of the ground-truth permutation.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PearsonLoss(nn.Module):
    """Differentiable Pearson correlation loss: 1 - pearson_r.

    Minimizing this loss maximizes the Pearson correlation between
    predictions and targets.

    Args:
        reduction: Reduction method — "mean", "sum", or "none".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation loss.

        Args:
            predictions: Model output scores, shape (N,).
            targets: Ground-truth values, shape (N,).

        Returns:
            Scalar loss value (1 - pearson_r), reduced according to self.reduction.
        """
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        num_samples = predictions.shape[0]

        pred_mean = predictions.mean()
        target_mean = targets.mean()

        covariance = (predictions * targets).sum() - pred_mean * target_mean * num_samples

        pred_variance = (predictions * predictions).sum() - pred_mean * pred_mean * num_samples
        target_variance = (targets * targets).sum() - target_mean * target_mean * num_samples

        pred_std = torch.sqrt(pred_variance.clamp(min=1e-8))
        target_std = torch.sqrt(target_variance.clamp(min=1e-8))

        denominator = pred_std * target_std + 1e-8
        pearson_r = covariance / denominator

        loss = 1 - pearson_r

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ListNetLoss(nn.Module):
    """ListNet loss for learning-to-rank.

    ListNet uses a softmax-based probability distribution and cross-entropy loss.
    For each group, items are ranked by their target values, then the loss is
    computed as the cross-entropy between predicted probabilities and target probabilities.

    Args:
        reduction: Reduction method — "mean", "sum", or "none".
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            group_indices: torch.Tensor = None,
            num_groups: int = None,
    ) -> torch.Tensor:
        """Compute ListNet loss.

        Args:
            predictions: Predicted probabilities after softmax, shape (N,).
            targets: Target probabilities (soft labels), shape (N,).
            group_indices: Group membership index for each sample, shape (N,); None for single-group mode.
            num_groups: Total number of distinct groups; required when group_indices is provided.

        Returns:
            Cross-entropy loss value.
        """
        if group_indices is not None and num_groups is not None:
            return self._forward_grouped(
                predictions=predictions, targets=targets,
                group_indices=group_indices, num_groups=num_groups,
            )
        else:
            return self._forward_single(predictions=predictions, targets=targets)

    def _forward_single(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ListNet loss for a single group.

        Args:
            predictions: Predicted probabilities after softmax, shape (N,).
            targets: Target probabilities (soft labels), shape (N,).

        Returns:
            Cross-entropy loss for this single group.
        """
        predictions = predictions.clamp(min=1e-12, max=1.0)
        targets = targets.clamp(min=1e-12, max=1.0)

        sample_losses = -targets * torch.log(predictions)
        total_loss = sample_losses.sum()

        if self.reduction == "mean":
            return total_loss / predictions.shape[0]
        elif self.reduction == "sum":
            return total_loss
        return total_loss

    def _forward_grouped(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            group_indices: torch.Tensor,
            num_groups: int,
    ) -> torch.Tensor:
        """Compute ListNet loss for grouped data using scatter operations.

        Args:
            predictions: Predicted probabilities after softmax, shape (N,).
            targets: Target probabilities (soft labels), shape (N,).
            group_indices: Group membership index for each sample, shape (N,).
            num_groups: Total number of distinct groups.

        Returns:
            Average cross-entropy loss across all groups.
        """
        predictions = predictions.clamp(min=1e-12, max=1.0)
        targets = targets.clamp(min=1e-12, max=1.0)

        sample_losses = -targets * torch.log(predictions)

        group_losses = torch.zeros(num_groups, device=predictions.device)
        group_counts = torch.zeros(num_groups, dtype=torch.long, device=predictions.device)

        group_losses.scatter_add_(0, group_indices, sample_losses)
        group_counts.scatter_add_(0, group_indices, torch.ones_like(group_indices, dtype=torch.long))

        valid_groups = group_counts > 0
        if valid_groups.any():
            if self.reduction == "mean":
                return group_losses[valid_groups].sum() / valid_groups.sum()
            else:
                return group_losses[valid_groups].sum()
        else:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)


class ListMLELoss(nn.Module):
    """List Maximum Likelihood Estimation loss for learning-to-rank.

    ListMLE maximizes the likelihood of the ground-truth permutation.
    For each group, items are sorted by their target values (descending),
    and the loss is the negative log-likelihood of this permutation under
    the predicted scores.

    Key difference from ListNet:
    - ListNet: Uses softmax probabilities directly with cross-entropy.
    - ListMLE: Uses conditional permutation probability, more sensitive to ranking order.

    Args:
        reduction: Reduction method — "mean", "sum", or "none".
        padded_value: Value used for padding masked positions (should be very negative).
    """

    def __init__(self, reduction: str = "mean", padded_value: float = -1e8) -> None:
        super().__init__()
        self.reduction = reduction
        self.padded_value = padded_value

    def forward(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            group_indices: torch.Tensor = None,
            num_groups: int = None,
    ) -> torch.Tensor:
        """Compute ListMLE loss.

        Args:
            predictions: Raw prediction scores (before softmax), shape (N,).
                         Should be the output before softmax for numerical stability.
            targets: Target values for ranking (higher is better), shape (N,).
            group_indices: Group membership index for each sample, shape (N,); None for single-group mode.
            num_groups: Total number of distinct groups; required when group_indices is provided.

        Returns:
            ListMLE loss value.
        """
        if group_indices is not None and num_groups is not None:
            return self._forward_grouped(
                predictions=predictions, targets=targets, group_indices=group_indices,
            )
        else:
            return self._forward_single(predictions=predictions, targets=targets)

    def _forward_single(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ListMLE loss for a single group/list using logcumsumexp.

        Args:
            predictions: Raw prediction scores, shape (N,).
            targets: Target values for ranking, shape (N,).

        Returns:
            ListMLE loss for this single group.
        """
        num_items = predictions.shape[0]
        if num_items <= 1:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        sorted_indices = torch.argsort(targets, descending=True)
        pred_sorted_by_target = predictions[sorted_indices]

        reversed_pred = torch.flip(pred_sorted_by_target, [0])
        log_cumsum_exp = torch.flip(torch.logcumsumexp(reversed_pred, dim=0), [0])

        total_loss = (log_cumsum_exp - pred_sorted_by_target).sum()

        if self.reduction == "mean":
            return total_loss / num_items
        elif self.reduction == "sum":
            return total_loss
        return total_loss

    def _forward_grouped(
            self,
            predictions: torch.Tensor,
            targets: torch.Tensor,
            group_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ListMLE loss for grouped data using pure 1D operations.

        This implementation avoids 2D padding and data-dependent shapes,
        making it fully compatible with torch.compile.

        Args:
            predictions: Raw prediction scores, shape (N,).
            targets: Target values for ranking, shape (N,).
            group_indices: Group membership index for each sample, shape (N,).

        Returns:
            Average ListMLE loss across all valid groups (size > 1).
        """
        if group_indices.numel() == 0:
            return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        group_indices = group_indices.to(predictions.device)
        device = predictions.device

        _, inverse_group_indices = torch.unique(group_indices, return_inverse=True)
        group_counts = torch.bincount(inverse_group_indices)
        nonzero_group_counts = group_counts[group_counts > 0]

        target_range = targets.max() - targets.min() + 1
        composite_sort_key = inverse_group_indices.float() * target_range - targets
        sort_order = torch.argsort(composite_sort_key, stable=True)

        pred_sorted = predictions[sort_order]
        sorted_inverse_groups = inverse_group_indices[sort_order]

        cumulative_counts = torch.cumsum(nonzero_group_counts, dim=0)
        group_start_positions = cumulative_counts - nonzero_group_counts

        exp_pred_sorted = torch.exp(pred_sorted)
        suffix_sum_exp = torch.cumsum(exp_pred_sorted.flip(0), dim=0).flip(0)
        suffix_sum_padded = torch.cat([suffix_sum_exp, torch.zeros(1, device=device)])

        group_end_positions = group_start_positions[sorted_inverse_groups] + nonzero_group_counts[
            sorted_inverse_groups]
        end_suffix_values = suffix_sum_padded[group_end_positions]
        log_partition = torch.log((suffix_sum_exp - end_suffix_values).clamp(min=1e-12))

        element_losses = log_partition - pred_sorted

        per_element_group_size = nonzero_group_counts[sorted_inverse_groups]
        valid_mask = per_element_group_size > 1

        total_loss = element_losses[valid_mask].sum()
        num_valid_groups = (nonzero_group_counts > 1).sum().float()

        if num_valid_groups == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss / num_valid_groups if self.reduction == "mean" else total_loss


def get_loss_function(loss_type: str = "pearson") -> nn.Module:
    """Factory function to create a loss function by name.

    Args:
        loss_type: One of "pearson", "listnet", "listmle".

    Returns:
        Instantiated loss module.

    Raises:
        ValueError: If loss_type is not recognized.
    """
    loss_type = loss_type.lower()

    if loss_type == "pearson":
        return PearsonLoss()
    elif loss_type == "listnet":
        return ListNetLoss()
    elif loss_type == "listmle":
        return ListMLELoss()
    else:
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Available: pearson, listnet, listmle"
        )


_COMPILABLE_LOSSES = {"pearson", "listnet", "listmle"}


def is_loss_compilable(loss_type: str) -> bool:
    """Check if the loss function can be safely compiled by torch.compile.

    Args:
        loss_type: Loss function name to check.

    Returns:
        True if the loss function is compilable, False otherwise.
    """
    return loss_type.lower() in _COMPILABLE_LOSSES
