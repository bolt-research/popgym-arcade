"""
Plot POMDP return, observability gap, and memory bias.

Usage:
    python plotting/return_gap_bias.py \
        --input-csv model_group.csv \
        --output-pdf PQN_gap_bias_plot.pdf

Optional:
    --no-show    Save the figure without opening a window.
    --no-usetex  Disable LaTeX text rendering.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot POMDP return, observability gap, and memory bias."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="model_group.csv",
        help="Input CSV produced by plotting/plottable.py. Default: model_group.csv",
    )
    parser.add_argument(
        "--output-pdf",
        type=str,
        default="PQN_gap_bias_plot.pdf",
        help="Output figure path. Default: PQN_gap_bias_plot.pdf",
    )
    parser.add_argument(
        "--show",
        dest="show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    parser.add_argument(
        "--no-show",
        dest="show",
        action="store_false",
        help="Save the figure without displaying it.",
    )
    parser.set_defaults(show=True)
    parser.add_argument(
        "--usetex",
        dest="usetex",
        action="store_true",
        help="Enable LaTeX text rendering.",
    )
    parser.add_argument(
        "--no-usetex",
        dest="usetex",
        action="store_false",
        help="Disable LaTeX text rendering.",
    )
    parser.set_defaults(usetex=True)
    return parser.parse_args()


def configure_plot_style(use_tex):
    sns.set_theme(style="white", context="talk")
    plt.rcParams["text.usetex"] = use_tex
    plt.rcParams["axes.labelsize"] = 22
    plt.rcParams["axes.titlesize"] = 22
    plt.rcParams["axes.grid"] = True
    plt.rcParams["axes.grid.axis"] = "y"
    plt.rcParams["grid.color"] = "#dbdbdb"


def load_and_prepare_data(input_csv):
    df = pd.read_csv(input_csv)

    print(df)
    df["Algorithm"] = (
        df["Algorithm"]
        .str.replace(r"PQN_RNN \(", "", regex=True)
        .str.replace(r"\)", "", regex=True)
    )
    df["Algorithm"] = df["Algorithm"].str.replace(r"PQN \(MLP", "MLP", regex=True)
    df["Algorithm"] = df["Algorithm"].str.replace("Gru", "GRU")
    df["Algorithm"] = df["Algorithm"].str.replace("Fart", "LAttn")
    df["Algorithm"] = df["Algorithm"].str.replace("Lru", "LRU")
    df["Algorithm"] = df["Algorithm"].str.replace("Mingru", "MinGRU")

    df["std"] = df["std"].fillna(0)
    df["sem"] = df["std"] / np.sqrt(df["count"])

    pivoted = df.pivot_table(
        index="Algorithm",
        columns="Partial",
        values=["mean", "sem"],
    )
    pivoted.columns = [f"{val}_{col}" for val, col in pivoted.columns]
    pivoted.reset_index(inplace=True)

    pivoted["gap"] = pivoted["mean_False"] - pivoted["mean_True"]
    pivoted["gap_sem"] = np.sqrt(pivoted["sem_False"] ** 2 + pivoted["sem_True"] ** 2)

    # Bias = J(pi, M) - J(f, pi, M). Positive bias means the recurrent model is worse on MDP.
    mlp_mdp_mean = pivoted.loc[pivoted["Algorithm"] == "MLP", "mean_False"].iloc[0]
    mlp_mdp_sem = pivoted.loc[pivoted["Algorithm"] == "MLP", "sem_False"].iloc[0]

    pivoted["bias"] = pivoted["mean_False"] - mlp_mdp_mean
    pivoted["bias_sem"] = np.sqrt(mlp_mdp_sem ** 2 + pivoted["sem_False"] ** 2)

    mlp_index = pivoted[pivoted["Algorithm"] == "MLP"].index
    pivoted.loc[mlp_index, "bias"] = 0
    pivoted.loc[mlp_index, "bias_sem"] = 0
    human_index = pivoted[pivoted["Algorithm"] == "Human"].index
    pivoted.loc[human_index, "bias"] = 0
    pivoted.loc[human_index, "bias_sem"] = 0

    algo_order = ["MLP", "Human", "MinGRU", "LAttn", "GRU", "LRU"]
    pivoted["Algorithm"] = pd.Categorical(
        pivoted["Algorithm"], categories=algo_order, ordered=True
    )
    plot_data = pivoted.sort_values("Algorithm")
    print(plot_data)
    return plot_data


def build_figure(plot_data):
    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5), sharey=False)

    algorithms = plot_data["Algorithm"]
    x_ind = np.arange(len(algorithms))
    shadow_offset = 0.08
    colors = ["#1a5225", "#257535", "#339c4c", "#4eb668", "#70cd88", "#95e1aa", "#bcf2cc"]

    ax1 = axes[0]
    ax1.bar(
        x_ind + shadow_offset,
        plot_data["mean_True"],
        width=0.8,
        color="black",
        alpha=0.3,
        zorder=1,
    )
    ax1.bar(
        x_ind,
        plot_data["mean_True"],
        yerr=plot_data["sem_True"],
        capsize=15,
        width=0.8,
        color=colors,
        alpha=0.8,
        zorder=2,
    )
    ax1.set_xticks(x_ind)
    ax1.set_xticklabels(algorithms)
    ax1.set_title("POMDP Return")
    ax1.set_ylim(bottom=0)

    ax2 = axes[1]
    ax2.bar(
        x_ind + shadow_offset,
        plot_data["gap"],
        width=0.8,
        color="black",
        alpha=0.3,
        zorder=1,
    )
    ax2.bar(
        x_ind,
        plot_data["gap"],
        yerr=plot_data["gap_sem"],
        capsize=15,
        width=0.8,
        color=colors,
        alpha=0.8,
        zorder=2,
    )
    ax2.set_xticks(x_ind)
    ax2.set_xticklabels(algorithms)
    ax2.set_title("Observability Gap")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")

    ax3 = axes[2]
    ax3.bar(
        x_ind + shadow_offset,
        plot_data["bias"],
        width=0.8,
        color="black",
        alpha=0.3,
        zorder=1,
    )
    ax3.bar(
        x_ind,
        plot_data["bias"],
        yerr=plot_data["bias_sem"],
        capsize=15,
        width=0.8,
        color=colors,
        alpha=0.8,
        zorder=2,
    )
    ax3.set_xticks(x_ind)
    ax3.set_xticklabels(algorithms)
    ax3.set_title("Memory Bias")
    ax3.axhline(0, color="black", linewidth=0.8, linestyle="--")

    sns.despine(fig=fig)
    plt.tight_layout()
    return fig


def main():
    args = parse_args()
    configure_plot_style(args.usetex)
    plot_data = load_and_prepare_data(args.input_csv)
    build_figure(plot_data)
    plt.savefig(args.output_pdf)
    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
