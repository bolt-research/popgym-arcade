#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}'

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

MODEL_TYPES: List[str] = ["fart", "gru", "lru", "mingru"]


def _find_csvs(env_name: str, memory_type: str, partial: bool, saliency_dir: str) -> List[str]:
    pattern = os.path.join(
        saliency_dir,
        f"saliency_results_{memory_type}_{env_name}_Partial={partial}_MODELSEED=*.csv",
    )
    return sorted(glob.glob(pattern))


def _extract_pos_columns(df: pd.DataFrame) -> np.ndarray:
    pos_cols = [c for c in df.columns if c.startswith("pos_")]
    # sort by numeric suffix
    pos_cols.sort(key=lambda c: float(c.split("pos_")[-1]))
    return df[pos_cols].to_numpy(dtype=float)


def _thirds_from_distribution_rows(pos_values: np.ndarray) -> np.ndarray:
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


def _aggregate_thirds(env_name: str, memory_type: str, partial: bool, saliency_dir: str) -> np.ndarray:
    files = _find_csvs(env_name, memory_type, partial, saliency_dir)
    if not files:
        return np.zeros(3, dtype=float)
    thirds_all = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue
        pos_vals = _extract_pos_columns(df)
        thirds_rows = _thirds_from_distribution_rows(pos_vals)
        if thirds_rows.size == 0:
            continue
        thirds_all.append(thirds_rows)
    if not thirds_all:
        return np.zeros(3, dtype=float)
    thirds_concat = np.concatenate(thirds_all, axis=0)
    mean_thirds = thirds_concat.mean(axis=0)
    s = mean_thirds.sum()
    return mean_thirds / s if s > 0 else np.zeros(3, dtype=float)


def plot_env(env_name: str, saliency_dir: str, output_dir: str, dpi: int = 300):
    mdp = np.vstack([
        _aggregate_thirds(env_name, m, False, saliency_dir) for m in MODEL_TYPES
    ])  # [4,3]
    pomdp = np.vstack([
        _aggregate_thirds(env_name, m, True, saliency_dir) for m in MODEL_TYPES
    ])

    mdp_colors = ["#C6DBEF", "#6BAED6", "#2171B5"]
    pomdp_colors = ["#FDD0A2", "#FDAE6B", "#E6550D"]

    # Slightly shorter length than before
    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1])

    x = np.arange(len(MODEL_TYPES))
    width = 0.6

    # MDP
    left = np.zeros(len(MODEL_TYPES))
    for k in range(3):
        ax_left.bar(x, mdp[:, k], width, bottom=left, color=mdp_colors[k], edgecolor="white", linewidth=0.6)
        left += mdp[:, k]
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([m.upper() for m in MODEL_TYPES], rotation=0, fontsize=12)
    ax_left.set_ylim(0, 1.05)
    ax_left.set_ylabel(r"$\mathbb{E}_{\pi, f}[\,\delta(Q_{\xi}(\mathbf{x},\tau))\,]$", fontsize=12)
    ax_left.set_title("MDP", fontsize=14)
    ax_left.spines["top"].set_visible(False)
    ax_left.spines["right"].set_visible(False)

    # POMDP
    left = np.zeros(len(MODEL_TYPES))
    for k in range(3):
        ax_right.bar(x, pomdp[:, k], width, bottom=left, color=pomdp_colors[k], edgecolor="white", linewidth=0.6)
        left += pomdp[:, k]
    ax_right.set_xticks(x)
    ax_right.set_xticklabels([m.upper() for m in MODEL_TYPES], rotation=0, fontsize=12)
    ax_right.set_ylim(0, 1.05)
    ax_right.set_title("POMDP", fontsize=14)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)

    # Legends (thirds)
    thirds_labels = [r"$[0,\frac{1}{3})$", r"$[\frac{1}{3},\frac{2}{3})$", r"$[\frac{2}{3},1)$"]
    legend_handles_left = [plt.Rectangle((0, 0), 1, 1, color=c) for c in mdp_colors]
    legend_handles_right = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pomdp_colors]
    
    anchor_x = 1.02
    ax_left.legend(
        legend_handles_left,
        thirds_labels,
        title="MDP thirds",
        loc="center right",
        bbox_to_anchor=(anchor_x, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        borderaxespad=0.0,
        labelspacing=0.4,
        handletextpad=0.6,
    )
    ax_right.legend(
        legend_handles_right,
        thirds_labels,
        title="POMDP thirds",
        loc="center left",
        bbox_to_anchor=(-anchor_x + 0.0, 0.5),
        fontsize=10,
        frameon=True,
        fancybox=True,
        borderaxespad=0.0,
        labelspacing=0.4,
        handletextpad=0.6,
    )

    fig.suptitle(env_name.replace("Easy", ""), fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"saliency_by_models_{env_name}.png")
    out_pdf = os.path.join(output_dir, f"saliency_by_models_{env_name}.pdf")
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_model(model_type: str, saliency_dir: str, output_dir: str, dpi: int = 300):
    mdp_colors = ["#C6DBEF", "#6BAED6", "#2171B5"]
    pomdp_colors = ["#FDD0A2", "#FDAE6B", "#E6550D"]

    out_dir = os.path.join(output_dir, f"by_model_{model_type}")
    os.makedirs(out_dir, exist_ok=True)

    for env_name in ENV_LIST:
        mdp = _aggregate_thirds(env_name, model_type, False, saliency_dir)
        pomdp = _aggregate_thirds(env_name, model_type, True, saliency_dir)

        # Shorter figure length for per-model-per-env plots
        fig, ax = plt.subplots(figsize=(6.5, 4))
        x = np.arange(2)
        width = 0.6

        # MDP stacked
        left = 0.0
        for k in range(3):
            ax.bar(x[0], mdp[k], width, bottom=left, color=mdp_colors[k], edgecolor="white", linewidth=0.6)
            left += mdp[k]
        # POMDP stacked
        left = 0.0
        for k in range(3):
            ax.bar(x[1], pomdp[k], width, bottom=left, color=pomdp_colors[k], edgecolor="white", linewidth=0.6)
            left += pomdp[k]

        ax.set_xticks(x)
        ax.set_xticklabels(["MDP", "POMDP"], fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r"$\mathbb{E}_{\pi, f}[\,\delta(Q_{\xi}(\mathbf{x},\tau))\,]$", fontsize=12)
        ax.set_title(f"{env_name.replace('Easy','')} — {model_type.upper()}", fontsize=14)
        # Remove top/right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        thirds_labels = [r"$[0,\frac{1}{3})$", r"$[\frac{1}{3},\frac{2}{3})$", r"$[\frac{2}{3},1)$"]
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=mdp_colors[0]),
            plt.Rectangle((0, 0), 1, 1, color=mdp_colors[1]),
            plt.Rectangle((0, 0), 1, 1, color=mdp_colors[2]),
            plt.Rectangle((0, 0), 1, 1, color=pomdp_colors[0]),
            plt.Rectangle((0, 0), 1, 1, color=pomdp_colors[1]),
            plt.Rectangle((0, 0), 1, 1, color=pomdp_colors[2]),
        ]
        legend_labels = [
            "MDP " + thirds_labels[0],
            "MDP " + thirds_labels[1],
            "MDP " + thirds_labels[2],
            "POMDP " + thirds_labels[0],
            "POMDP " + thirds_labels[1],
            "POMDP " + thirds_labels[2],
        ]
        ax.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=9, frameon=True, fancybox=True)

        fig.tight_layout(rect=[0, 0, 1, 0.92])

        out_png = os.path.join(out_dir, f"{model_type}_{env_name}.png")
        out_pdf = os.path.join(out_dir, f"{model_type}_{env_name}.pdf")
        fig.savefig(out_png, dpi=dpi)
        fig.savefig(out_pdf)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot per-model figures: for each model, 10 env figures with MDP vs POMDP")
    parser.add_argument("--saliency_dir", type=str, default="your_saliency_csv_dir")
    parser.add_argument("--output_dir", type=str, default="your_output_dir")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--models", type=str, default=",".join(MODEL_TYPES), help="Comma-separated model types to include (fart,gru,lru,mingru)")
    args = parser.parse_args()

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in selected_models:
        plot_model(m, args.saliency_dir, args.output_dir, dpi=args.dpi)
        print(f"Saved 10 figs for model: {m}")


if __name__ == "__main__":
    main()


