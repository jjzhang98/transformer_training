"""Visualization utilities for model training and evaluation results.

All functions accept an optional ``save_path``. When provided the figure is
saved (and closed); otherwise it is shown interactively.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # non-interactive backend; safe for server / scripts

from .metrics import TOPK_VALUES

_PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
_TOPK_COLORS = ["#E91E63", "#9C27B0", "#3F51B5", "#009688"]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, path: Optional[str], dpi: int = 150) -> None:
    if path:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Training curve
# ---------------------------------------------------------------------------

def plot_training_curve(
    history: dict,
    title: str = "Training History",
    save_path: Optional[str] = None,
) -> None:
    """Plot training curves including loss, hit rates, and new topk metrics.

    ``history`` dict can contain any subset of:
        train_loss, val_loss,
        train_topk_hit_rate, val_topk_hit_rate,
        train_excess_hit_rate, val_excess_hit_rate,
        train_avg_pred_rank, val_avg_pred_rank,
        train_top1_above_mean_rate, val_top1_above_mean_rate,
        train_top{k}_above_mean_rate, val_top{k}_above_mean_rate for k in [1,3,5,10],
        train_top{k}_vs_prior_rate, val_top{k}_vs_prior_rate for k in [1,3,5,10]
    """
    has_loss = "train_loss" in history or "val_loss" in history
    has_topk = "train_topk_hit_rate" in history or "val_topk_hit_rate" in history
    has_excess = "train_excess_hit_rate" in history or "val_excess_hit_rate" in history
    has_pred_rank = "train_avg_pred_rank" in history or "val_avg_pred_rank" in history

    has_topk_above_mean = any(
        f"val_top{k}_above_mean_rate" in history or f"train_top{k}_above_mean_rate" in history
        for k in TOPK_VALUES
    )
    has_topk_vs_prior = any(
        f"val_top{k}_vs_prior_rate" in history or f"train_top{k}_vs_prior_rate" in history
        for k in TOPK_VALUES
    )

    n_plots = sum([has_loss, has_topk, has_excess, has_pred_rank,
                   has_topk_above_mean, has_topk_vs_prior])

    if n_plots == 0:
        return

    n_cols = min(n_plots, 4)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    panel = 0

    if has_loss:
        ax = axes[panel]
        if "train_loss" in history:
            ax.plot(history["train_loss"], label="Train Loss", color=_PALETTE[0], linewidth=1.5, alpha=0.8)
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Val Loss", color=_PALETTE[1], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        panel += 1

    if has_topk:
        ax = axes[panel]
        if "train_topk_hit_rate" in history:
            ax.plot(history["train_topk_hit_rate"], label="Train Top-K Hit Rate",
                    color=_PALETTE[0], linewidth=1.5, alpha=0.8)
        if "val_topk_hit_rate" in history:
            ax.plot(history["val_topk_hit_rate"], label="Val Top-K Hit Rate",
                    color=_PALETTE[1], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Hit Rate")
        ax.set_title("Top-K Hit Rate")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim([0, 1])
        panel += 1

    if has_excess:
        ax = axes[panel]
        if "train_excess_hit_rate" in history:
            ax.plot(history["train_excess_hit_rate"], label="Train Excess Hit Rate",
                    color=_PALETTE[0], linewidth=1.5, alpha=0.8)
        if "val_excess_hit_rate" in history:
            ax.plot(history["val_excess_hit_rate"], label="Val Excess Hit Rate",
                    color=_PALETTE[1], linewidth=1.5)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Excess Hit Rate")
        ax.set_title("Excess Hit Rate (vs Baseline)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        panel += 1

    if has_pred_rank:
        ax = axes[panel]
        if "train_avg_pred_rank" in history:
            ax.plot(history["train_avg_pred_rank"], label="Train Avg Pred Rank",
                    color=_PALETTE[0], linewidth=1.5, alpha=0.8)
        if "val_avg_pred_rank" in history:
            ax.plot(history["val_avg_pred_rank"], label="Val Avg Pred Rank",
                    color=_PALETTE[1], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Pred Rank")
        ax.set_title("Average Prediction Rank")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim([0, 1])
        panel += 1

    if has_topk_above_mean:
        ax = axes[panel]
        for idx, k in enumerate(TOPK_VALUES):
            val_key = f"val_top{k}_above_mean_rate"
            if val_key in history:
                ax.plot(history[val_key], label=f"Val Top{k}",
                        color=_TOPK_COLORS[idx % len(_TOPK_COLORS)], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rate")
        ax.set_title("TopK Above Mean Rate")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim([0, 1])
        panel += 1

    if has_topk_vs_prior:
        ax = axes[panel]
        for idx, k in enumerate(TOPK_VALUES):
            val_key = f"val_top{k}_vs_prior_rate"
            if val_key in history:
                ax.plot(history[val_key], label=f"Val Top{k} vs Prior",
                        color=_TOPK_COLORS[idx % len(_TOPK_COLORS)], linewidth=1.5)
        ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Rate")
        ax.set_title("TopK vs Prior Abtype Rate")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim([0, 1])
        panel += 1

    for i in range(panel, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance: pd.Series,
    title: str = "Feature Importance",
    top_n: int = 30,
    save_path: Optional[str] = None,
) -> None:
    """Horizontal bar chart of feature importances (top N).

    Args:
        importance: pd.Series indexed by feature name, values = importance score.
    """
    if importance is None or importance.empty:
        return

    imp = importance.sort_values(ascending=True).tail(top_n)
    n = len(imp)

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.35)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, n))
    ax.barh(np.arange(n), imp.values, color=colors)
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(imp.index, fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"{title} (top {n})")
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# Prediction scatter / density
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    title: str = "Predictions vs Actuals",
    save_path: Optional[str] = None,
) -> None:
    """Hexbin density plot of predictions vs actual labels."""
    y_pred = np.asarray(y_pred).ravel()
    y_true = np.asarray(y_true).ravel()
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    y_pred, y_true = y_pred[mask], y_true[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(y_true, y_pred, gridsize=50, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Count")
    # diagonal reference line
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y=x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _savefig(fig, save_path)


# ---------------------------------------------------------------------------
# Abtype distribution comparison
# ---------------------------------------------------------------------------

def plot_abtype_distribution(
        df: pd.DataFrame,
        y_pred: np.ndarray,
        y_col: str,
        title: str = "Abtype Distribution: True vs Predicted",
        save_path: Optional[str] = None,
) -> None:
    """Compare distribution of abtypes selected by max true label vs max prediction.

    For each (date, code, time_sec) group:
        - Find abtype with max true label
        - Find abtype with max prediction
    Then plot the distribution of selected abtypes for both cases.
    """
    required_cols = ['date', 'code', 'time_sec', 'abtype', y_col]
    if not all(col in df.columns for col in required_cols):
        print(f"  [Warning] Missing required columns for abtype distribution plot")
        return

    if len(df) != len(y_pred):
        print(f"  [Warning] DataFrame length {len(df)} != prediction length {len(y_pred)}")
        return

    df_work = df.copy().reset_index(drop=True)
    df_work['_pred'] = y_pred


    grouped = df_work.groupby(['date', 'code', 'time_sec'])
    true_max_idx = grouped[y_col].idxmax()
    pred_max_idx = grouped['_pred'].idxmax()

    # 过滤空组或全 NaN 组
    valid_mask = true_max_idx.notna() & pred_max_idx.notna()
    if not valid_mask.any():
        print(f"  [Warning] No valid groups found for abtype distribution plot")
        return

    true_abtypes = df_work.loc[true_max_idx[valid_mask], 'abtype'].values
    pred_abtypes = df_work.loc[pred_max_idx[valid_mask], 'abtype'].values

    if len(true_abtypes) == 0:
        print(f"  [Warning] No valid groups found for abtype distribution plot")
        return

    all_abtypes = sorted(set(true_abtypes) | set(pred_abtypes))

    true_counts = pd.Series(true_abtypes).value_counts().reindex(all_abtypes, fill_value=0)
    pred_counts = pd.Series(pred_abtypes).value_counts().reindex(all_abtypes, fill_value=0)

    # 归一化为百分比
    true_props = true_counts / true_counts.sum() * 100
    pred_props = pred_counts / pred_counts.sum() * 100

    x = np.arange(len(all_abtypes))
    width = 0.35
    n_abtypes = len(all_abtypes)

    # 🎨 动态画布宽度：按类别数线性扩展，限制上下限防止过小或过大
    fig_width = min(32, max(10, n_abtypes * 1.0))

    # 📏 智能标签排版策略（全部显示，通过旋转+字号+对齐方式防重叠）
    if n_abtypes <= 15:
        rotation, fontsize, ha = 45, 9, 'right'
    elif n_abtypes <= 30:
        rotation, fontsize, ha = 90, 8, 'center'
    elif n_abtypes <= 50:
        rotation, fontsize, ha = 90, 7, 'center'
    else:
        rotation, fontsize, ha = 90, 6, 'center'

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars1 = ax.bar(x - width / 2, true_props.values, width, label='True (Max Label)',
                   color=_PALETTE[0], alpha=0.8)
    bars2 = ax.bar(x + width / 2, pred_props.values, width, label='Predicted (Max Pred)',
                   color=_PALETTE[1], alpha=0.8)

    ax.set_xlabel('Abtype')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.set_xticks(x)

    # 标签格式化：float -> str(int)，确保全量显示
    labels = [str(int(ab)) for ab in all_abtypes]
    ax.set_xticklabels(labels, rotation=rotation, ha=ha, fontsize=fontsize)

    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 柱顶百分比标注（字号略调小，避免与 x 轴标签碰撞）
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=7)

    # 自动紧凑布局 + 垂直标签底部留白补偿
    fig.tight_layout()
    if rotation == 90:
        fig.subplots_adjust(bottom=0.22)

    _savefig(fig, save_path)
