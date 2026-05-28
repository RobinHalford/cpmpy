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


_ALGO_DISPLAY = {
    "marco":                "MARCO",
    "marco_diverse_shrink": "D-MARCO-DGS",
    "marco_diverse_solhint":"D-MARCO-HINT",
    "marco_diverse_min":    "D-MARCO-COUNT",
    "marco_diverse_opt":    "D-MARCO-EXACT",
    "ocus_enum1":           "D-OCUS-SW",
    "ocus_enum_shrink":     "D-OCUS-CW",
    "ocus_enum_opt":        "D-OCUS-EXACT",
    "marco_select_topk":    "MARCO-TOPK",
}

_ALGO_COLORS = {
    "marco":                "#d62728",
    "marco_diverse_shrink": "#ff7f0e",
    "marco_diverse_solhint":"#bcbd22",
    "marco_diverse_min":    "#2ca02c",
    "marco_diverse_opt":    "#17becf",
    "ocus_enum1":           "#1f77b4",
    "ocus_enum_shrink":     "#9467bd",
    "ocus_enum_opt":        "#e377c2",
    "marco_select_topk":    "#8c564b",
}

def avg_diversity_gain_vs_marco(input_csv: Path, output_csv: Path) -> None:
    """For each algorithm, compute the average diversity gain over marco for k=2..10.

    For each (instance, k) pair where both the algorithm and marco produced at
    least k MUSes, compute diversity(algo first k MUSes) - diversity(marco first k
    MUSes).  Average these differences across all qualifying instances for each k,
    then write one row per algorithm to output_csv.
    """
    df = pd.read_csv(input_csv)

    # Parse MUSes once per row
    df["_muses"] = df["MUSes"].apply(lambda x: parse_mus_list(str(x)))
    df["_n"] = df["_muses"].apply(len)

    # Build per-instance marco lookup: instance -> list of MUSes (up to 10)
    marco_df = df[df["algorithm"] == "marco"].set_index("instance")

    algorithms = [a for a in df["algorithm"].unique() if a != "marco"]
    ks = list(range(2, 11))

    rows = []
    for algo in algorithms:
        algo_df = df[df["algorithm"] == algo].set_index("instance")
        sums = {k: 0.0 for k in ks}
        counts = {k: 0 for k in ks}

        for instance, algo_row in algo_df.iterrows():
            if instance not in marco_df.index:
                continue
            marco_row = marco_df.loc[instance]
            algo_muses = algo_row["_muses"]
            marco_muses = marco_row["_muses"]

            for k in ks:
                if len(algo_muses) < k or len(marco_muses) < k:
                    continue
                algo_div = float(diversity_setOfMUSes(algo_muses[:k]))
                marco_div = float(diversity_setOfMUSes(marco_muses[:k]))
                sums[k] += algo_div - marco_div
                counts[k] += 1

        row = {"algorithm": algo}
        for k in ks:
            row[f"avg_div_gain_{k}"] = round(sums[k] / counts[k], 6) if counts[k] > 0 else None
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=["algorithm"] + [f"avg_div_gain_{k}" for k in ks])
    out_df.to_csv(output_csv, index=False)


def avg_diversity_vs_k(input_csv: Path, output_csv: Path) -> None:
    """For each algorithm, compute the average diversity of the first k MUSes for k=2..10.

    For each (instance, k) pair where the algorithm produced at least k MUSes,
    compute diversity(algo first k MUSes).  Average these across all qualifying
    instances for each k, then write one row per algorithm to output_csv.
    """
    df = pd.read_csv(input_csv)

    df["_muses"] = df["MUSes"].apply(lambda x: parse_mus_list(str(x)))

    ks = list(range(2, 11))

    rows = []
    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo]
        sums = {k: 0.0 for k in ks}
        counts = {k: 0 for k in ks}

        for _, row in algo_df.iterrows():
            muses = row["_muses"]
            for k in ks:
                if len(muses) < k:
                    continue
                sums[k] += float(diversity_setOfMUSes(muses[:k]))
                counts[k] += 1

        result_row = {"algorithm": algo}
        for k in ks:
            result_row[f"avg_div_{k}"] = round(sums[k] / counts[k], 6) if counts[k] > 0 else None
        rows.append(result_row)

    out_df = pd.DataFrame(rows, columns=["algorithm"] + [f"avg_div_{k}" for k in ks])
    out_df.to_csv(output_csv, index=False)


def avg_diversity_topk(input_csvs: List[Path], output_csv: Path) -> None:
    """For each k, compute the average diversity from the top-k selection files.

    Rows where n < k are excluded (the algorithm did not find k MUSes).
    One output row per k with the average diversity across qualifying instances.
    """
    df = pd.concat([pd.read_csv(p) for p in input_csvs], ignore_index=True)

    rows = []
    for k, group in df.groupby("k"):
        qualifying = group[group["n"] >= k]["diversity"].dropna()
        rows.append({
            "k": k,
            "avg_diversity": round(qualifying.mean(), 6) if len(qualifying) > 0 else None,
            "num_instances": len(qualifying),
        })

    pd.DataFrame(rows).sort_values("k").to_csv(output_csv, index=False)


def avg_diversity_gain_topk_vs_marco(input_csvs: List[Path], marco_csv: Path, output_csv: Path) -> None:
    """For each k, compute the average diversity gain of top-k selection over marco.

    For each (instance, k) pair where both top-k selection has n >= k and marco
    has at least k MUSes, compute diversity(top-k) - diversity(marco first k MUSes).
    Average these across qualifying instances per k.
    """
    topk_df = pd.concat([pd.read_csv(p) for p in input_csvs], ignore_index=True)
    topk_df = topk_df[topk_df["n"] >= topk_df["k"]].dropna(subset=["diversity"])

    marco_df = pd.read_csv(marco_csv)
    marco_df = marco_df[marco_df["algorithm"] == "marco"].copy()
    marco_df["_muses"] = marco_df["MUSes"].apply(lambda x: parse_mus_list(str(x)))
    marco_lookup = marco_df.set_index("instance")["_muses"].to_dict()

    rows = []
    for k, group in topk_df.groupby("k"):
        gains = []
        for _, row in group.iterrows():
            instance = row["instance"]
            if instance not in marco_lookup:
                continue
            marco_muses = marco_lookup[instance]
            if len(marco_muses) < k:
                continue
            marco_div = float(diversity_setOfMUSes(marco_muses[:k]))
            gains.append(row["diversity"] - marco_div)

        rows.append({
            "k": k,
            "avg_diversity_gain": round(float(np.mean(gains)), 6) if gains else None,
            "num_instances": len(gains),
        })

    pd.DataFrame(rows).sort_values("k").to_csv(output_csv, index=False)


def plot_avg_diversity_vs_k(avg_div_csv: Path, avg_div_topk_csv: Path, output_dir: Path) -> None:
    """Plot average diversity vs k for all algorithms and marco-select-top-k.

    x-axis: k in {2, 3, 4, 5, 10}; y-axis: average diversity in [0, 1].
    Saved as a PDF suitable for inclusion in LaTeX documents.
    """
    ks = [2, 3, 4, 5, 10]

    # Load per-algorithm diversity (columns avg_div_2 .. avg_div_10)
    div_df = pd.read_csv(avg_div_csv)
    algo_data: dict = {}
    for _, row in div_df.iterrows():
        algo = row["algorithm"]
        vals = [row.get(f"avg_div_{k}") for k in ks]
        algo_data[algo] = vals

    # Load top-k selection diversity (rows keyed by k)
    topk_df = pd.read_csv(avg_div_topk_csv).set_index("k")
    algo_data["marco_select_topk"] = [
        topk_df.loc[k, "avg_diversity"] if k in topk_df.index else None
        for k in ks
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in _ALGO_DISPLAY:
        if algo not in algo_data:
            continue
        vals = algo_data[algo]
        # Only plot k values that have a non-null value
        plot_ks = [k for k, v in zip(ks, vals) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        plot_vs = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not plot_ks:
            continue
        color = _ALGO_COLORS.get(algo, "#333333")
        label = _ALGO_DISPLAY[algo]
        ax.plot(plot_ks, plot_vs, marker="o", markersize=4, linewidth=1.5,
                color=color, label=label)

    ax.set_xlim(1.5, 10.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(ks)
    ax.set_xlabel("k (number of MUSes)", fontsize=11)
    ax.set_ylabel("average diversity", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "avg_diversity_vs_k.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_avg_diversity_vs_k_allcomplete(avg_div_csv: Path, output_dir: Path) -> None:
    """Plot average diversity vs k=2..5 over the all-complete instance set.

    Uses average_diversity_vs_k_ALLCOMPLETE.csv (no top-k column, k limited to
    2-5).  Saved as avg_diversity_vs_k_ALLCOMPLETE.pdf in output_dir.
    """
    ks = [2, 3, 4, 5]

    div_df = pd.read_csv(avg_div_csv)
    algo_data: dict = {}
    for _, row in div_df.iterrows():
        algo = row["algorithm"]
        algo_data[algo] = [row.get(f"avg_div_{k}") for k in ks]

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo in _ALGO_DISPLAY:
        if algo not in algo_data:
            continue
        vals = algo_data[algo]
        plot_ks = [k for k, v in zip(ks, vals) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        plot_vs = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not plot_ks:
            continue
        ax.plot(plot_ks, plot_vs, marker="o", markersize=4, linewidth=1.5,
                color=_ALGO_COLORS.get(algo, "#333333"), label=_ALGO_DISPLAY[algo])

    ax.set_xlim(1.5, 5.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(ks)
    ax.set_xlabel("k (number of MUSes)", fontsize=11)
    ax.set_ylabel("average diversity", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir / "avg_diversity_vs_k_ALLCOMPLETE.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def avg_time_to_k5_allcomplete(input_csv: Path, output_csv: Path) -> None:
    """Average time to find the 5th MUS, restricted to the all-complete instance set.

    The all-complete set is the same 20 instances used by avg_diversity_vs_k_allcomplete:
    every non-topk algorithm must have found at least 5 MUSes.  Time to k=5 is the
    5th cumulative runtime timestamp (runtimes[4]).
    """
    df = pd.read_csv(input_csv)
    df = df[df["algorithm"] != "marco_select_topk"].copy()
    df["_muses"] = df["MUSes"].apply(lambda x: parse_mus_list(str(x)))
    df["_n"] = df["_muses"].apply(len)
    df["_runtimes"] = df["runtimes"].apply(lambda x: parse_runtime_list(str(x)))

    min_n_per_instance = df.groupby("instance")["_n"].min()
    complete_instances = min_n_per_instance[min_n_per_instance >= 5].index
    df = df[df["instance"].isin(complete_instances)]

    rows = []
    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo]
        times = [row["_runtimes"][4] for _, row in algo_df.iterrows() if len(row["_runtimes"]) >= 5]
        rows.append({
            "algorithm": algo,
            "avg_time_to_k5": round(float(np.mean(times)), 6) if times else None,
            "num_instances": len(times),
        })

    out_df = pd.DataFrame(rows, columns=["algorithm", "avg_time_to_k5", "num_instances"])
    out_df = out_df.sort_values("avg_time_to_k5")
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote avg time to k=5 for {len(complete_instances)} instances to {output_csv}")


def diversity_drop_table(input_csv: Path, output_csv: Path) -> None:
    """For each algorithm, compute the drop in average diversity between k=2→5 and k=5→10.

    For each transition (k_low → k_high), only instances where the algorithm found at
    least k_high MUSes are included.  The drop is avg_div(k_low) - avg_div(k_high) on
    that filtered set, so a positive value means diversity fell as k increased.

    Columns written: algorithm, avg_div_2, avg_div_5 (on instances with ≥5 MUSes),
    drop_2to5, n_2to5, avg_div_5_for10, avg_div_10 (on instances with ≥10 MUSes),
    drop_5to10, n_5to10.
    """
    df = pd.read_csv(input_csv)
    df["_muses"] = df["MUSes"].apply(lambda x: parse_mus_list(str(x)))

    transitions = [(2, 5), (5, 10)]
    rows = []

    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo]
        row: dict = {"algorithm": algo}

        for k_low, k_high in transitions:
            qualifying = [r["_muses"] for _, r in algo_df.iterrows() if len(r["_muses"]) >= k_high]
            n = len(qualifying)
            if n > 0:
                avg_low = round(sum(float(diversity_setOfMUSes(m[:k_low])) for m in qualifying) / n, 6)
                avg_high = round(sum(float(diversity_setOfMUSes(m[:k_high])) for m in qualifying) / n, 6)
                drop = round(avg_low - avg_high, 6)
            else:
                avg_low = avg_high = drop = None

            label = f"{k_low}to{k_high}"
            row[f"avg_div_{k_low}_for{label}"] = avg_low
            row[f"avg_div_{k_high}_for{label}"] = avg_high
            row[f"drop_{label}"] = drop
            row[f"n_{label}"] = n

        rows.append(row)

    columns = ["algorithm"]
    for k_low, k_high in transitions:
        label = f"{k_low}to{k_high}"
        columns += [f"avg_div_{k_low}_for{label}", f"avg_div_{k_high}_for{label}", f"drop_{label}", f"n_{label}"]

    pd.DataFrame(rows, columns=columns).to_csv(output_csv, index=False)


def avg_diversity_vs_k_allcomplete(input_csv: Path, output_csv: Path) -> None:
    """Average diversity for k=2..5 restricted to instances where ALL algorithms have >=5 MUSes.

    Excludes marco_select_topk (post-hoc selection, not comparable).  For each
    remaining instance, only includes it if every algorithm present for that
    instance found at least 5 MUSes.  Averages diversity(first k MUSes) over
    that filtered set for k in {2, 3, 4, 5}.
    """
    df = pd.read_csv(input_csv)
    df = df[df["algorithm"] != "marco_select_topk"].copy()
    df["_muses"] = df["MUSes"].apply(lambda x: parse_mus_list(str(x)))
    df["_n"] = df["_muses"].apply(len)

    # Keep only instances where every algorithm reached >=5 MUSes
    min_n_per_instance = df.groupby("instance")["_n"].min()
    complete_instances = min_n_per_instance[min_n_per_instance >= 5].index
    df = df[df["instance"].isin(complete_instances)]

    ks = [2, 3, 4, 5]
    rows = []
    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo]
        sums = {k: 0.0 for k in ks}
        counts = {k: 0 for k in ks}

        for _, row in algo_df.iterrows():
            muses = row["_muses"]
            for k in ks:
                if len(muses) < k:
                    continue
                sums[k] += float(diversity_setOfMUSes(muses[:k]))
                counts[k] += 1

        result_row = {"algorithm": algo}
        for k in ks:
            result_row[f"avg_div_{k}"] = round(sums[k] / counts[k], 6) if counts[k] > 0 else None
        rows.append(result_row)

    out_df = pd.DataFrame(rows, columns=["algorithm"] + [f"avg_div_{k}" for k in ks])
    out_df.to_csv(output_csv, index=False)
    print(f"Wrote {len(complete_instances)} complete instances to {output_csv}")


def plot_time_vs_diversity_k(input_csv: Path, output_dir: Path, k: int) -> None:
    """Scatter plot: avg time to k-th MUS (x) vs avg diversity at k (y).

    Only instances where every algorithm (excluding marco_select_topk) found at
    least k MUSes are included.  Each algorithm is rendered as a single dot with
    a legend.
    """
    df = pd.read_csv(input_csv)
    df = df[df["algorithm"] != "marco_select_topk"].copy()
    df["_muses"] = df["MUSes"].apply(lambda x: parse_mus_list(str(x)))
    df["_n"] = df["_muses"].apply(len)
    df["_runtimes"] = df["runtimes"].apply(lambda x: parse_runtime_list(str(x)))

    min_n_per_instance = df.groupby("instance")["_n"].min()
    complete_instances = min_n_per_instance[min_n_per_instance >= k].index
    df = df[df["instance"].isin(complete_instances)]

    points = {}
    for algo in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algo]
        times, divs = [], []
        for _, row in algo_df.iterrows():
            if len(row["_runtimes"]) >= k and len(row["_muses"]) >= k:
                times.append(row["_runtimes"][k - 1])
                divs.append(float(diversity_setOfMUSes(row["_muses"][:k])))
        if times:
            points[algo] = (float(np.mean(times)), float(np.mean(divs)))

    fig, ax = plt.subplots(figsize=(6, 4))

    for algo, (avg_time, avg_div) in points.items():
        color = _ALGO_COLORS.get(algo, "#333333")
        label = _ALGO_DISPLAY.get(algo, algo)
        ax.scatter(avg_time, avg_div, color=color, s=60, zorder=3, label=label)

    ax.set_xlabel(f"average time to k={k} (s)", fontsize=11)
    ax.set_ylabel(f"average diversity at k={k}", fontsize=11)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.6)

    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"time_vs_diversity_k{k}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote scatter plot ({len(complete_instances)} instances) to {out_path}")


def main() -> None:
    input_csv = Path("results/xcsp_20260525_161136.csv")
    results_dir = Path("results")
    plots_dir = results_dir / "plots"

    topk_csvs = sorted(results_dir.glob("xcsp_marco_select_top_k_*.csv"))

    avg_diversity_vs_k(input_csv, results_dir / "average_diversity_vs_k.csv")
    avg_diversity_gain_vs_marco(input_csv, results_dir / "average_diversity_gain_vs_marco.csv")
    diversity_drop_table(input_csv, results_dir / "diversity_drop.csv")

    allcomplete_csv = results_dir / "average_diversity_vs_k_ALLCOMPLETE.csv"
    avg_diversity_vs_k_allcomplete(input_csv, allcomplete_csv)
    avg_time_to_k5_allcomplete(input_csv, results_dir / "average_time_to_k5_ALLCOMPLETE.csv")

    plot_avg_diversity_vs_k(
        results_dir / "average_diversity_vs_k.csv",
        results_dir / "average_diversity_topk.csv",
        plots_dir,
    )
    plot_avg_diversity_vs_k_allcomplete(allcomplete_csv, plots_dir)
    plot_time_vs_diversity_k(input_csv, plots_dir, k=5)
    plot_time_vs_diversity_k(input_csv, plots_dir, k=10)


if __name__ == "__main__":
    main()
