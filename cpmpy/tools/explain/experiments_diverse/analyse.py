from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import List, Sequence
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from cpmpy.tools.explain.utils import diversity_setOfMUSes, diversity_matrix
from cpmpy.tools.explain.visualize_diversity import visualize_heatmap


def parse_runtime_list(text: str) -> List[float]:
    """Parse a string like '[0.1, 0.3, 0.9]' into a list of floats."""
    text = (text or "").strip()
    if not text or text == "[]":
        return []

    values: List[float] = []
    for part in text.strip("[]").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            # Skip malformed runtime entries rather than crashing.
            continue
    return values


def parse_mus_list(text: str) -> List[List[int]]:
    """
    Parse a MUS collection string like "[[0, 2, 5], [1, 3]]" into a list of
    index lists.  MUSes are stored as constraint indices (into model.constraints)
    rather than raw constraint objects.
    """
    text = (text or "").strip()
    if not text or text == "[]":
        return []
    return ast.literal_eval(text)


def total_runtime(runtimes: Sequence[float]) -> float:
    """Treat runtimes as cumulative timestamps and use the last one."""
    return runtimes[-1] if runtimes else 0.0


def parse_diversity_curve(text: str) -> List[tuple]:
    """Parse a diversity curve string like '[(0.1, 0.0), (0.5, 0.3)]' into a list of (timestamp, min_div) tuples."""
    text = (text or "").strip()
    if not text or text == "[]":
        return []
    return ast.literal_eval(text)


def filter_completed_xcsp(input_csv: Path, output_csv: Path) -> None:
    """From the xcsp results file, keep only rows for instances where every
    algorithm completed (status == 'COMPLETE').

    Adds a ``diversity_curve`` column — a list of (runtime, min_diversity) tuples,
    one per MUS, where min_diversity at position i is the minimum pairwise
    diversity over the first i+1 MUSes (0 for the first MUS).

    The ``runtimes`` and ``MUSes`` columns are dropped from the output.
    """
    df = pd.read_csv(input_csv)

    # Keep only instances where every algorithm completed
    completed = df.groupby("instance")["status"].apply(lambda s: (s == "COMPLETE").all())
    completed_instances = completed[completed].index
    df = df[df["instance"].isin(completed_instances)].copy()

    def _build_curve(row):
        muses = parse_mus_list(str(row["MUSes"]))
        runtimes = parse_runtime_list(str(row["runtimes"]))
        curve = []
        for i, t in enumerate(runtimes):
            if i == 0:
                div = 0.0
            else:
                div = float(diversity_setOfMUSes(muses[:i + 1]))
            curve.append((t, div))
        return curve

    df["diversity_curve"] = df.apply(_build_curve, axis=1)
    df["min_diversity"] = df["diversity_curve"].apply(lambda c: c[-1][1] if c else 0.0)
    df = df.drop(columns=["runtimes", "MUSes"])
    df = df.sort_values(["instance", "algorithm"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def filter_completed_xcsp_topk(input_csv: Path, output_csv: Path) -> None:
    """From the xcsp_topk results file, keep only rows where the algorithm
    completed (status == 'COMPLETE'), and add a min_diversity column.
    """
    df = pd.read_csv(input_csv)
    df = df[df["status"] == "COMPLETE"].copy()

    def _min_div(row):
        curve = parse_diversity_curve(str(row["diversity_curve"]))
        return curve[-1][1] if curve else 0.0

    df["min_diversity"] = df.apply(_min_div, axis=1)
    df = df.sort_values(["instance", "algorithm"]).reset_index(drop=True)
    df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


_ALGO_DISPLAY = {
    "marco":               "MARCO",
    "marco_diverse_noMin": "D-MARCO v1",
    "marco_diverse_min":   "D-MARCO v2",
    "marco_diverse_opt":   "D-MARCO v3",
    "ocus_enum1":          "D-OCUS v1",
    "ocus_enum_shrink":    "D-OCUS v2",
    "ocus_enum_opt":       "D-OCUS v3",
    "marco_until_diverse": "MARCO top k",
}

_ALGO_COLORS = {
    "marco":               "#f71919",
    "marco_diverse_noMin": "#e78631",
    "marco_diverse_min":   "#beb50f",
    "marco_diverse_opt":   "#ff04c9",
    "ocus_enum1":          "#55ae47",
    "ocus_enum_shrink":    "#8a03a8",
    "ocus_enum_opt":       "#10e892",
    "marco_until_diverse": "#0b53c8",
}


def plot_anytime_curves(all_csv: Path, topk_csv: Path, output_dir: Path) -> None:
    """Merge completed_all and completed_topk on common instances and produce
    one anytime-diversity PNG per instance saved into output_dir.

    The x-axis is log-scaled time; the y-axis is diversity fixed to [0, 1].
    Each point in the diversity_curve is plotted as a marker and curves are
    drawn as step functions (post-step) to reflect the anytime nature.
    """
    df_all  = pd.read_csv(all_csv)
    df_topk = pd.read_csv(topk_csv)

    common = set(df_all["instance"]) & set(df_topk["instance"])
    df_all  = df_all[df_all["instance"].isin(common)]
    df_topk = df_topk[df_topk["instance"].isin(common)]

    combined = pd.concat([df_all, df_topk], ignore_index=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for instance, group in combined.groupby("instance"):
        fig, ax = plt.subplots(figsize=(8, 5))

        plotted = {}  # algo -> scatter handle for legend

        # Draw marco_until_diverse first so it sits in the background
        for _, row in group.iterrows():
            if row["algorithm"] != "marco_until_diverse":
                continue
            curve = parse_diversity_curve(str(row["diversity_curve"]))
            if not curve:
                continue
            times = [t for t, _ in curve]
            divs  = [d for _, d in curve]
            color = _ALGO_COLORS["marco_until_diverse"]
            ax.plot(times, divs, color=color, linewidth=1.0, alpha=0.4, zorder=1)
            sc = ax.scatter(times, divs, color=color, s=40, zorder=2)
            plotted["marco_until_diverse"] = sc

        # Draw all other algorithms on top — last point only
        for _, row in group.iterrows():
            algo = row["algorithm"]
            if algo == "marco_until_diverse":
                continue
            curve = parse_diversity_curve(str(row["diversity_curve"]))
            if not curve:
                continue
            t, d = curve[-1]
            color = _ALGO_COLORS.get(algo, None)
            sc = ax.scatter([t], [d], color=color, s=60, zorder=3)
            plotted[algo] = sc

        ax.set_xscale("log")
        ax.set_xlim(1e-2, 60)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Time log(s)")
        ax.set_ylabel("diversity of set of MUSes")

        # Legend ordered by _ALGO_DISPLAY, only for algorithms that were plotted
        ordered_handles = []
        ordered_labels  = []
        for algo, label in _ALGO_DISPLAY.items():
            if algo in plotted:
                ordered_handles.append(plotted[algo])
                ordered_labels.append(label)
        ax.legend(ordered_handles, ordered_labels, loc="lower right", fontsize=8)

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

        safe_name = str(instance).replace("/", "_").replace("\\", "_")
        fig.savefig(output_dir/"plots"/f"{safe_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MUS result CSV.")
    parser.add_argument("input_csv", type=Path, help="Path to the original CSV file")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory")
    args = parser.parse_args()

    filter_completed_xcsp(args.input_csv,  args.output_dir/"completed_all.csv")
    filter_completed_xcsp_topk(args.output_dir/"xcsp_topk_20260416_165301.csv", args.output_dir/"completed_topk.csv")

    plot_anytime_curves(args.output_dir/"completed_all.csv", args.output_dir/"completed_topk.csv", args.output_dir)


if __name__ == "__main__":
    main()
