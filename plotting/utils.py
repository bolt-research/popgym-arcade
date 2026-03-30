#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd


EASY_ENV_MAX_STEPS = {
    # Classic
    "CartPoleEasy": 200,
    "NoisyCartPoleEasy": 200,
    # Memory games
    "CountRecallEasy": 126,  # 100 + 26
    "AutoEncodeEasy": 260,   # 26 * 1 * 2 * 5
    # Gridworlds
    "NavigatorEasy": 64,     # 8 * 8
    "BattleShipEasy": 128,   # 8 * 8 * 2
    "MineSweeperEasy": 32,   # 4 * 4 * 2
    # Arcade-style
    "BreakoutEasy": 2000,
    "SkittlesEasy": 100,
    "TetrisEasy": 3000,
}


@dataclass
class RecallDensityResult:
    seed: int
    distribution: np.ndarray
    dist_path: str

    @property
    def length(self) -> int:
        return int(len(self.distribution))


def easy_max_steps_for_env(env_name: str) -> int:
    """Return Easy-difficulty max steps for known environments."""
    return EASY_ENV_MAX_STEPS.get(env_name, 200)


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def parse_seeds_arg(seeds_arg: str) -> list[int]:
    """Accept formats like '0,1,2,3,4', '0..4', or '0'."""
    if ".." in seeds_arg:
        start, end = seeds_arg.split("..", 1)
        return list(range(int(start), int(end) + 1))
    if "," in seeds_arg:
        return [int(seed.strip()) for seed in seeds_arg.split(",") if seed.strip()]
    return [int(seeds_arg)]


def collect_pkl_files(root: str) -> Iterator[tuple[str, str]]:
    """Recursively yield (file_dir, filename) for every .pkl under root."""
    for dirpath, _, files in os.walk(root):
        for filename in sorted(files):
            if filename.endswith(".pkl"):
                yield dirpath, filename


def algorithm_label_from_prefix(prefix: str) -> str:
    """Map model filename prefixes like PQN_RNN to output labels like pqn."""
    return prefix.split("_", 1)[0].lower()


def save_recall_density_csv(
    results: Sequence[RecallDensityResult],
    env_name: str,
    output_csv: str,
    max_steps: Optional[int] = None,
) -> str:
    """Save per-seed recall-density results to a padded CSV table."""
    if not results:
        raise ValueError("No recall-density results to save.")

    max_length = int(max_steps) if max_steps is not None else easy_max_steps_for_env(env_name)
    rows = []
    for result in results:
        padded_dist = np.zeros(max_length, dtype=float)
        upto = min(result.length, max_length)
        padded_dist[:upto] = result.distribution[:upto]

        row = {
            "seed": result.seed,
            "length": result.length,
            "dist_path": result.dist_path,
        }
        for index in range(max_length):
            norm_pos = index / max_length if max_length > 0 else 0.0
            row[f"pos_{norm_pos:.3f}"] = padded_dist[index]
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return output_csv


def parse_saliency_csv_filename(filename: str):
    """Parse generated saliency CSV names to recover env and partial."""
    pattern = (
        r"^saliency_results_[^_]+_[^_]+_(?P<env>.+?)_Partial="
        r"(?P<partial>True|False)(?:_.*)?\.csv$"
    )
    match = re.match(pattern, filename)
    if not match:
        return None
    return {
        "ENV_NAME": match.group("env"),
        "PARTIAL": match.group("partial") == "True",
    }


def extract_pos_columns(df: pd.DataFrame) -> np.ndarray:
    pos_cols = [column for column in df.columns if column.startswith("pos_")]
    pos_cols.sort(key=lambda column: float(column.split("pos_")[-1]))
    return df[pos_cols].to_numpy(dtype=float)


def thirds_from_distribution_rows(pos_values: np.ndarray) -> np.ndarray:
    """Convert per-position density rows into three normalized thirds."""
    if pos_values.size == 0:
        return np.zeros((pos_values.shape[0], 3), dtype=float)

    num_cols = pos_values.shape[1]
    edge1 = num_cols // 3
    edge2 = (num_cols * 2) // 3
    thirds = np.stack(
        [
            pos_values[:, :edge1].sum(axis=1),
            pos_values[:, edge1:edge2].sum(axis=1),
            pos_values[:, edge2:].sum(axis=1),
        ],
        axis=1,
    )
    row_sums = thirds.sum(axis=1, keepdims=True)
    normalized = np.zeros_like(thirds, dtype=float)
    valid = row_sums[:, 0] > 0
    if np.any(valid):
        normalized[valid] = thirds[valid] / row_sums[valid]
    return normalized


def build_saliency_bar_data(saliency_dir: str) -> pd.DataFrame:
    """Aggregate generated saliency CSVs into the stacked-bar values used for plotting."""
    csv_paths = sorted(glob.glob(os.path.join(saliency_dir, "saliency_results_*.csv")))
    grouped_rows: dict[tuple[str, bool], list[np.ndarray]] = {}
    source_csv_counts: dict[tuple[str, bool], int] = {}
    source_seed_counts: dict[tuple[str, bool], int] = {}

    for path in csv_paths:
        meta = parse_saliency_csv_filename(os.path.basename(path))
        if meta is None:
            continue

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[warn] Failed to read {path}: {exc}")
            continue

        if df.empty:
            continue

        pos_values = extract_pos_columns(df)
        thirds_rows = thirds_from_distribution_rows(pos_values)
        if thirds_rows.size == 0:
            continue

        key = (meta["ENV_NAME"], meta["PARTIAL"])
        grouped_rows.setdefault(key, []).append(thirds_rows)
        source_csv_counts[key] = source_csv_counts.get(key, 0) + 1
        source_seed_counts[key] = source_seed_counts.get(key, 0) + len(df)

    summary_rows = []
    for (env_name, partial), thirds_chunks in sorted(grouped_rows.items()):
        thirds_concat = np.concatenate(thirds_chunks, axis=0)
        mean_thirds = thirds_concat.mean(axis=0)
        mean_sum = mean_thirds.sum()
        thirds = mean_thirds / mean_sum if mean_sum > 0 else np.zeros(3, dtype=float)
        summary_rows.append(
            {
                "EnvName": env_name,
                "Partial": partial,
                "third_1": thirds[0],
                "third_2": thirds[1],
                "third_3": thirds[2],
                "source_csv_count": source_csv_counts[(env_name, partial)],
                "source_seed_count": source_seed_counts[(env_name, partial)],
            }
        )

    return pd.DataFrame(summary_rows)


def save_saliency_bar_data(saliency_dir: str, output_csv: str) -> Optional[str]:
    """Save the aggregated stacked-bar data used by plot_saliency_summary.py."""
    summary_df = build_saliency_bar_data(saliency_dir)
    if summary_df.empty:
        print(f"[warn] No saliency CSVs available to summarize under: {saliency_dir}")
        return None

    summary_df.to_csv(output_csv, index=False)
    print(f"Bar-summary data saved to {output_csv}")
    return output_csv
