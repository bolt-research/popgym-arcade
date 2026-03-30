#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re


ENV_LIST: List[str] = [
    "AutoEncodeEasy",
    "BattleShipEasy",
    "BreakoutEasy",
    "CartPoleEasy",
    "CountRecallEasy",
    "MineSweeperEasy",
    "NavigatorEasy",
    "NoisyCartPoleEasy",
    "SkittlesEasy",
    "TetrisEasy",
]


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'

def _find_csvs_for(env_name: str, partial: bool, saliency_dir: str) -> List[str]:
    pattern = os.path.join(
        saliency_dir, f"saliency_results_*_{env_name}_Partial={partial}_*.csv"
    )
    return sorted(glob.glob(pattern))


def _extract_pos_columns(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    # Sort by numeric suffix
    def _key(c: str) -> float:
        try:
            return float(c.split("pos_")[-1])
        except Exception:
            return 0.0

    pos_cols.sort(key=_key)
    return df[pos_cols].to_numpy(dtype=float), pos_cols


def _thirds_from_distribution_rows(pos_values: np.ndarray) -> np.ndarray:
    """Given matrix [num_rows, num_positions], compute thirds per row.

    Returns array [num_rows, 3] with per-third sums normalized so each row sums to 1
    (if a row sums to 0, it remains zeros).
    """
    if pos_values.size == 0:
        return np.zeros((pos_values.shape[0], 3), dtype=float)

    num_cols = pos_values.shape[1]
    e1 = num_cols // 3
    e2 = (num_cols * 2) // 3
    thirds = np.stack(
        [
            pos_values[:, :e1].sum(axis=1),
            pos_values[:, e1:e2].sum(axis=1),
            pos_values[:, e2:].sum(axis=1),
        ],
        axis=1,
    )
    row_sums = thirds.sum(axis=1, keepdims=True)
    norm = np.zeros_like(thirds, dtype=float)
    mask = row_sums[:, 0] > 0
    if np.any(mask):
        norm[mask] = thirds[mask] / row_sums[mask]
    return norm


def compute_env_mode_summary(env_name: str, partial: bool, saliency_dir: str) -> np.ndarray:
    """Aggregate all CSVs for (env, partial) across files and seeds.

    Returns a vector of length 3 that sums to 1 (or zeros if nothing found).
    """
    csv_files = _find_csvs_for(env_name, partial, saliency_dir)
    if not csv_files:
        return np.zeros(3, dtype=float)

    thirds_all = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        pos_values, _ = _extract_pos_columns(df)
        thirds_rows = _thirds_from_distribution_rows(pos_values)
        if thirds_rows.size == 0:
            continue
        thirds_all.append(thirds_rows)

    if not thirds_all:
        return np.zeros(3, dtype=float)

    thirds_concat = np.concatenate(thirds_all, axis=0)  # [num_total_rows, 3]
    mean_thirds = thirds_concat.mean(axis=0)
    mean_sum = mean_thirds.sum()
    return mean_thirds / mean_sum if mean_sum > 0 else np.zeros(3, dtype=float)


def _load_data_from_summary_csv(summary_csv: str) -> np.ndarray:
    """Load pre-aggregated bar data with columns EnvName, Partial, third_1..third_3."""
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"Summary CSV is empty: {summary_csv}")

    required_columns = {"EnvName", "Partial", "third_1", "third_2", "third_3"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Summary CSV is missing required columns: {sorted(missing)}"
        )

    partial_series = df["Partial"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
    )

    data = np.zeros((len(ENV_LIST), 2, 3), dtype=float)
    partial_modes = [False, True]
    for i, env in enumerate(ENV_LIST):
        for j, part in enumerate(partial_modes):
            match = df[(df["EnvName"] == env) & (partial_series == part)]
            if match.empty:
                continue
            row = match.iloc[0]
            thirds = np.array([row["third_1"], row["third_2"], row["third_3"]], dtype=float)
            total = thirds.sum()
            data[i, j] = thirds / total if total > 0 else np.zeros(3, dtype=float)
    return data


def _load_data_from_saliency_dir(saliency_dir: str) -> np.ndarray:
    """Aggregate raw saliency CSVs into plot-ready thirds data."""
    partial_modes = [False, True]
    data = np.zeros((len(ENV_LIST), len(partial_modes), 3), dtype=float)
    for i, env in enumerate(ENV_LIST):
        for j, part in enumerate(partial_modes):
            data[i, j] = compute_env_mode_summary(env, part, saliency_dir)
    return data


def _plot_mode(env_values: np.ndarray, colors: list[str], mode_title: str, output_path: str, dpi: int) -> None:

    fig, ax = plt.subplots(figsize=(min(20, 1.8 * len(ENV_LIST)), 7))
    x = np.arange(len(ENV_LIST))

    # Stacked thirds for single mode
    left = np.zeros(len(ENV_LIST))
    for k in range(3):
        values = env_values[:, k]
        ax.bar(
            x,
            values,
            0.6,
            bottom=left,
            color=colors[k],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        left += values

    # x ticks and fonts
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("Easy", "") for e in ENV_LIST], rotation=30, ha="right", fontsize=30)
    ax.tick_params(axis="both", labelsize=30)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"Saliency mass per third", fontsize=14)

    # Legend (thirds only)
    third_labels = [r"$[0,\frac{1}{3})$", r"$[\frac{1}{3},\frac{2}{3})$", r"$[\frac{2}{3},1)$"]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[0]),
        plt.Rectangle((0, 0), 1, 1, color=colors[1]),
        plt.Rectangle((0, 0), 1, 1, color=colors[2]),
    ]
    ax.legend(legend_handles, third_labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=30, frameon=True, fancybox=True, handlelength=1.2, handletextpad=0.6)

    ax.set_title(f"Aggregate Recall Density — {mode_title}", fontsize=40)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    
    # If the user didn't request a PDF explicitly, save a copy as PDF too
    base, ext = os.path.splitext(output_path)
    if ext.lower() != ".pdf":
        fig.savefig(base + ".pdf")
    plt.close(fig)


def _plot_bars_on_ax(ax, env_values: np.ndarray, colors: list[str], show_left_axis: bool, ylabel_math: str | None = None):
    x = np.arange(len(ENV_LIST))
    left = np.zeros(len(ENV_LIST))
    for k in range(3):
        vals = env_values[:, k]
        ax.bar(
            x,
            vals,
            0.6,
            bottom=left,
            color=colors[k],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
        )
        left += vals
    ax.set_xticks(x)
    # Use single-line labels 

    single_line_labels = [e.replace("Easy", "") for e in ENV_LIST]
    ax.set_xticklabels(single_line_labels, rotation=30, ha="right", fontsize=30)
    # ax.tick_params(axis="x", pad=15)
    ax.set_ylim(0, 1.05)
    # Beautify: remove top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_left_axis:
        ax.tick_params(axis="y", labelsize=35, left=True, labelleft=True)
        if ylabel_math:
            ax.set_ylabel(ylabel_math, fontsize=35, rotation=90, labelpad=10)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)


def plot_summary(
    output_path: str,
    dpi: int = 300,
    saliency_dir: Optional[str] = None,
    summary_csv: Optional[str] = None,
):
    if summary_csv is not None:
        data = _load_data_from_summary_csv(summary_csv)
    elif saliency_dir is not None:
        data = _load_data_from_saliency_dir(saliency_dir)
    else:
        raise ValueError("Provide either saliency_dir or summary_csv.")

    # Colors
    mdp_colors = ["#C6DBEF", "#6BAED6", "#2171B5"]
    pomdp_colors = ["#FDD0A2", "#FDAE6B", "#E6550D"]

    fig = plt.figure(figsize=(max(32, 1.7 * len(ENV_LIST) * 2), 8.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 0.9, 2.2], wspace=0.12)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_center = fig.add_subplot(gs[0, 1])
    ax_right = fig.add_subplot(gs[0, 2])

    ylabel_math = r"$\mathbb{E}_{\pi, f}[\,\delta(Q(\mathbf{x},\tau))\,]$"
    _plot_bars_on_ax(ax_left, data[:, 0, :], mdp_colors, show_left_axis=True, ylabel_math=ylabel_math)
    _plot_bars_on_ax(ax_right, data[:, 1, :], pomdp_colors, show_left_axis=False)

    # Center block: title + legend (combined 6 items)
    ax_center.axis("off")
    ax_center.set_title("Aggregate Recall Density", fontsize=40, pad=10)
    mdp_handles = [
        plt.Rectangle((0, 0), 1, 1, color=mdp_colors[0]),
        plt.Rectangle((0, 0), 1, 1, color=mdp_colors[1]),
        plt.Rectangle((0, 0), 1, 1, color=mdp_colors[2]),
    ]
    pomdp_handles = [
        plt.Rectangle((0, 0), 1, 1, color=pomdp_colors[0]),
        plt.Rectangle((0, 0), 1, 1, color=pomdp_colors[1]),
        plt.Rectangle((0, 0), 1, 1, color=pomdp_colors[2]),
    ]
    mdp_labels = [
        r"MDP $0<\tau<0.33$",
        r"MDP $0.33\leq\,\tau<0.66$",
        r"MDP $0.66\leq\,\tau<1.0$",
    ]
    pomdp_labels = [
        r"POMDP $0<\tau<0.33$",
        r"POMDP $0.33\leq\,\tau<0.66$",
        r"POMDP $0.66\leq\,\tau<1.0$",
    ]

    common_legend_kwargs = dict(frameon=True, fancybox=True, fontsize=30, handlelength=1.8, handletextpad=0.8)

    combined_handles = mdp_handles + pomdp_handles
    combined_labels = mdp_labels + pomdp_labels
    ax_center.legend(
        combined_handles,
        combined_labels,
        loc="center",
        bbox_to_anchor=(0.44, 0.5),
        ncol=1,
        borderaxespad=0.0,
        labelspacing=0.6,
        **common_legend_kwargs,
    )

    plt.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.92, wspace=0.12)
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    
    base, _ = os.path.splitext(output_path)
    if not output_path.lower().endswith(".pdf"):
        fig.savefig(base + ".pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated saliency thirds per env and mode")
    parser.add_argument(
        "--saliency_dir",
        type=str,
        default=None,
        help="Directory containing per-weight saliency CSVs",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default=None,
        help="Pre-aggregated bar-summary CSV generated by density_analysis_{pqn,ppo}.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="your_output_pdf",
        help="Path to save the summary figure (PNG & PDF)",
    )
    
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.summary_csv is None and args.saliency_dir is None:
        raise SystemExit("Please provide either --summary_csv or --saliency_dir")

    plot_summary(
        output_path=args.output,
        dpi=args.dpi,
        saliency_dir=args.saliency_dir,
        summary_csv=args.summary_csv,
    )
    print(f"Saved summary to {args.output}")

if __name__ == "__main__":
    main()