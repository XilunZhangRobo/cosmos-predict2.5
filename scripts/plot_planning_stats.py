#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Plot planning stats from saved *_planning_stats.json files.

Usage:
  python scripts/plot_planning_stats.py <path>
  python scripts/plot_planning_stats.py outputs/action_conditioned/basic_planned
  python scripts/plot_planning_stats.py outputs/action_conditioned/basic_planned/0_planning_stats.json

If <path> is a directory, finds all *_planning_stats.json inside and plots each (or overlay).
If <path> is a file, plots that single JSON.
Outputs are saved as <path>/planning_plots/ or next to the JSON file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_stats(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_episode(stats: dict, out_dir: Path, prefix: str = "") -> None:
    """Plot one episode's planning stats."""
    img_name = stats.get("img_name", "?")
    num_chunks = stats["num_chunks"]
    chunks = stats["chunks"]

    # 1) Cost convergence per chunk: elite and best cost over CEM iterations
    ncols = min(4, num_chunks)
    nrows = (num_chunks + ncols - 1) // ncols
    fig1, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if num_chunks == 1:
        axes = np.array([[axes]])
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1)
    for idx, ch in enumerate(chunks):
        r, c = idx // ncols, idx % ncols
        ax = axes[r, c]
        iters = np.arange(1, len(ch["elite_costs_per_iter"]) + 1)
        ax.plot(iters, ch["elite_costs_per_iter"], "o-", label="elite mean", markersize=4)
        ax.plot(iters, ch["best_costs_per_iter"], "s-", label="best", markersize=4)
        ax.set_xlabel("CEM iteration")
        ax.set_ylabel("Cost")
        ax.set_title(f"Chunk {ch['chunk_idx']} (goal frame {ch['goal_frame_idx']})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    for idx in range(num_chunks, nrows * ncols):
        r, c = idx // ncols, idx % ncols
        axes[r, c].set_visible(False)
    fig1.suptitle(f"Episode {img_name}: CEM convergence per chunk")
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{prefix}episode_{img_name}_cem_per_chunk.png", dpi=120)
    plt.close(fig1)

    # 2) Episode progress: final cost per chunk
    fig2, ax = plt.subplots(1, 1, figsize=(6, 4))
    chunk_idxs = [ch["chunk_idx"] for ch in chunks]
    elite = [ch["final_elite_mean"] for ch in chunks]
    best = [ch["final_best_cost"] for ch in chunks]
    ax.plot(chunk_idxs, elite, "o-", label="final elite mean", markersize=6)
    ax.plot(chunk_idxs, best, "s-", label="final best cost", markersize=6)
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Cost")
    ax.set_title(f"Episode {img_name}: cost at end of each chunk")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{prefix}episode_{img_name}_cost_per_chunk.png", dpi=120)
    plt.close(fig2)

    # 3) Action std per chunk (CEM spread) – last CEM iter only
    fig3, ax = plt.subplots(1, 1, figsize=(6, 4))
    stds = [ch["action_std_mean_per_iter"][-1] for ch in chunks]
    ax.bar(chunk_idxs, stds, color="steelblue", alpha=0.8)
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Action std (last CEM iter)")
    ax.set_title(f"Episode {img_name}: action uncertainty at convergence")
    ax.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    fig3.savefig(out_dir / f"{prefix}episode_{img_name}_action_std.png", dpi=120)
    plt.close(fig3)

    # 4) Ground-truth action error per chunk (if available)
    has_action_error = any("action_l2_mean" in ch for ch in chunks)
    if has_action_error:
        fig4, ax = plt.subplots(1, 1, figsize=(6, 4))
        action_error_means = [
            ch.get("action_l2_mean") if ch.get("action_l2_mean") is not None else float("nan") for ch in chunks
        ]
        ax.plot(chunk_idxs, action_error_means, "o-", markersize=6)
        ax.set_xlabel("Chunk index")
        ax.set_ylabel("Mean action L2 error")
        ax.set_title(f"Episode {img_name}: action error vs. ground truth actions")
        ax.grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(out_dir / f"{prefix}episode_{img_name}_action_error_per_chunk.png", dpi=120)
        plt.close(fig4)


def plot_all_episodes_overlay(stats_list: list[tuple[str, dict]], out_dir: Path) -> None:
    """Overlay final cost per chunk for all episodes."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for name, stats in stats_list:
        chunks = stats["chunks"]
        chunk_idxs = [ch["chunk_idx"] for ch in chunks]
        best = [ch["final_best_cost"] for ch in chunks]
        ax.plot(chunk_idxs, best, "o-", label=f"Episode {name}", markersize=4)
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Final best cost")
    ax.set_title("All episodes: best cost per chunk")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "all_episodes_cost_overlay.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        default=Path("outputs/action_conditioned/basic_planned"),
        nargs="?",
        help="Directory containing *_planning_stats.json or path to a single JSON file",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="When path is a dir, also plot one overlay figure for all episodes",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <path>/planning_plots or same dir as JSON)",
    )
    args = parser.parse_args()
    path = args.path

    if path.is_file():
        if not path.name.endswith(".json") or "_planning_stats" not in path.name:
            raise SystemExit("Single path must be a *_planning_stats.json file")
        stats_list = [(path.stem.replace("_planning_stats", ""), load_stats(path))]
        out_dir = args.output_dir or path.parent / "planning_plots"
    else:
        if not path.is_dir():
            raise SystemExit(f"Not a file or directory: {path}")
        json_files = sorted(path.glob("*_planning_stats.json"))
        if not json_files:
            raise SystemExit(f"No *_planning_stats.json found in {path}")
        stats_list = [
            (f.stem.replace("_planning_stats", ""), load_stats(f))
            for f in json_files
        ]
        out_dir = args.output_dir or path / "planning_plots"

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots to {out_dir}")

    for name, stats in stats_list:
        plot_episode(stats, out_dir, prefix="")

    if args.overlay and len(stats_list) > 1:
        plot_all_episodes_overlay(stats_list, out_dir)
        print("Saved overlay: all_episodes_cost_overlay.png")

    print("Done.")


if __name__ == "__main__":
    main()
