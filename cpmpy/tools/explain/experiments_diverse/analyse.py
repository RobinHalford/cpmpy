from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import List, Sequence
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


def diversity_per_instance(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    required_columns = {"instance", "algorithm", "status", "runtimes", "MUSes"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        muses = parse_mus_list(str(row["MUSes"]))
        runtimes = parse_runtime_list(str(row["runtimes"]))

        rows.append(
            {
                "instance": row["instance"],
                "algorithm": row["algorithm"],
                "status": row["status"],
                "num_muses": len(muses),
                "total_runtime": total_runtime(runtimes),
                "min_diversity": diversity_setOfMUSes(muses) if len(muses) > 1 else 0,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["instance", "algorithm"]).reset_index(drop=True)

    summary_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def compare_to_marco_per_instance(summary_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(summary_csv)

    required_cols = {
        "instance",
        "algorithm",
        "status",
        "total_runtime",
        "min_diversity",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Start with NaN for everything
    df["diversity_gain_vs_marco"] = np.nan
    df["additional_runtime_vs_marco"] = np.nan

    marco_mask = df["algorithm"].eq("marco")
    df.loc[marco_mask, ["diversity_gain_vs_marco", "additional_runtime_vs_marco"]] = 0.0

    marco_baseline = (
        df.loc[marco_mask & df["status"].eq("COMPLETE"), ["instance", "min_diversity", "total_runtime"]]
        .rename(
            columns={
                "min_diversity": "marco_min_diversity",
                "total_runtime": "marco_total_runtime",
            }
        )
    )

    df = df.merge(marco_baseline, on="instance", how="left")
    completed_non_marco = df["status"].eq("COMPLETE") & ~df["algorithm"].eq("marco")

    df.loc[completed_non_marco, "diversity_gain_vs_marco"] = (
        df.loc[completed_non_marco, "min_diversity"]
        - df.loc[completed_non_marco, "marco_min_diversity"]
    )

    df.loc[completed_non_marco, "additional_runtime_vs_marco"] = (
        df.loc[completed_non_marco, "total_runtime"]
        - df.loc[completed_non_marco, "marco_total_runtime"]
    )

    df = df.drop(columns=["marco_min_diversity", "marco_total_runtime"])

    df.to_csv(output_csv, index=False)


def summarize_per_algorithm(compare_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(compare_csv)

    required_cols = {
        "algorithm",
        "status",
        "diversity_gain_vs_marco",
        "additional_runtime_vs_marco",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    df = df[df["algorithm"] != "marco"].copy()
    timeout_pct = (df.groupby("algorithm")["status"].apply(lambda s: 100 * s.eq("TIMEOUT").mean()).rename("timeout_pct"))

    completed_instances = df.groupby("instance")["status"].apply(lambda s: s.eq("COMPLETE").all())

    df_all_completed = df[df["instance"].isin(completed_instances[completed_instances].index)]
    df_all_completed.to_csv(output_csv.parent / "completed_instances.csv", index=False)

    agg = (
        df_all_completed.groupby("algorithm")
        .agg(
            mean_diversity_gain_vs_marco=("diversity_gain_vs_marco", "mean"),
            avg_additional_runtime_vs_marco=("additional_runtime_vs_marco", "mean"),
        )
    )
    result = timeout_pct.to_frame().join(agg)
    result.to_csv(output_csv)


def parse_diversity_curve(text: str) -> List[tuple]:
    """Parse a diversity curve string like '[(0.1, 0.0), (0.5, 0.3)]' into a list of (timestamp, min_div) tuples."""
    text = (text or "").strip()
    if not text or text == "[]":
        return []
    return ast.literal_eval(text)


def topk_diversity_per_instance(input_csv: Path, output_csv: Path) -> None:
    """Summarise marco_until_diverse results from a diversity-curve CSV.

    Reads the ``diversity_curve`` column (list of ``(timestamp, best_min_div)``
    tuples) and derives the same output schema as ``diversity_per_instance``
    so that ``compare_to_topk_per_instance`` can consume it unchanged.
    """
    df = pd.read_csv(input_csv)

    required_columns = {"instance", "algorithm", "status", "diversity_curve"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows = []
    for _, row in df.iterrows():
        curve = parse_diversity_curve(str(row["diversity_curve"]))
        total_runtime = curve[-1][0] if curve else 0.0
        min_diversity = curve[-1][1] if curve else 0.0
        rows.append(
            {
                "instance": row["instance"],
                "algorithm": row["algorithm"],
                "status": row["status"],
                "num_muses": len(curve),
                "total_runtime": total_runtime,
                "min_diversity": min_diversity,
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["instance", "algorithm"]).reset_index(drop=True)
    summary_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def compare_to_topk_per_instance(
    diversity_csv: Path, topk_csv: Path, output_csv: Path) -> None:
    """Compare every algorithm in *diversity_csv* against the
    ``marco_until_diverse`` baseline found in *topk_csv*.

    Output columns
    --------------
    instance, algorithm, status, num_muses, total_runtime, min_diversity,
    diversity_marco_topk, outperforms_topk

    ``outperforms_topk`` is True when status == "COMPLETE" **and**
    min_diversity > diversity_marco_topk.
    """
    df = pd.read_csv(diversity_csv)
    topk_df = pd.read_csv(topk_csv)

    for name, src in [("diversity_csv", df), ("topk_csv", topk_df)]:
        required = {"instance", "algorithm", "status", "num_muses", "total_runtime", "min_diversity"}
        missing = required - set(src.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {sorted(missing)}")

    marco_topk = topk_df[topk_df["algorithm"] == "marco_until_diverse"][
        ["instance", "min_diversity", "status"]
    ].rename(columns={"min_diversity": "diversity_marco_topk", "status": "marco_topk_status"})

    df = df.merge(marco_topk, on="instance", how="left")

    marco_timed_out = df["marco_topk_status"].eq("TIMEOUT")
    df["outperforms_topk"] = df["status"].eq("COMPLETE") & (
        marco_timed_out | (df["min_diversity"] >= df["diversity_marco_topk"])
    )

    df = df.drop(columns=["marco_topk_status"])

    out_cols = [
        "instance", "algorithm", "status", "num_muses",
        "total_runtime", "min_diversity", "diversity_marco_topk", "outperforms_topk",
    ]
    df[out_cols].sort_values(["instance", "algorithm"]).reset_index(drop=True).to_csv(
        output_csv, index=False, quoting=csv.QUOTE_MINIMAL
    )


def summarize_topk_comparison(compare_topk_csv: Path, output_csv: Path) -> None:
    """Summarize how often each algorithm outperforms ``marco_until_diverse``.

    Output columns
    --------------
    algorithm, num_instances, num_outperforms, pct_outperforms
    """
    df = pd.read_csv(compare_topk_csv)

    required = {"algorithm", "instance", "outperforms_topk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    result = (
        df.groupby("algorithm")
        .agg(
            num_instances=("instance", "count"),
            num_outperforms=("outperforms_topk", "sum"),
        )
        .assign(pct_outperforms=lambda d: 100 * d["num_outperforms"] / d["num_instances"])
        .sort_values("pct_outperforms", ascending=False)
    )

    result.to_csv(output_csv, quoting=csv.QUOTE_MINIMAL)


def summarize_completed(
    completed_csv: Path, compare_topk_csv: Path, output_csv: Path
) -> None:
    """For instances where every algorithm completed, summarise per algorithm:

    - diversity_gain_vs_marco        : mean(min_diversity - marco min_diversity)
    - diversity_gain_vs_marco_topk   : mean(min_diversity - marco_until_diverse min_diversity)
    - additional_runtime_vs_marco    : mean(total_runtime  - marco total_runtime)

    All three metrics can be negative.
    """
    completed = pd.read_csv(completed_csv)
    topk = pd.read_csv(compare_topk_csv)

    for name, df, required in [
        ("completed_csv", completed, {"instance", "algorithm", "min_diversity", "diversity_gain_vs_marco", "additional_runtime_vs_marco"}),
        ("compare_topk_csv", topk,   {"instance", "diversity_marco_topk"}),
    ]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {name}: {sorted(missing)}")

    # One diversity_marco_topk value per instance (same across all algorithms)
    topk_baseline = topk[["instance", "diversity_marco_topk"]].drop_duplicates("instance")

    df = completed.merge(topk_baseline, on="instance", how="left")
    df["diversity_gain_vs_marco_topk"] = df["min_diversity"] - df["diversity_marco_topk"]

    result = (
        df[df["algorithm"] != "marco"]
        .groupby("algorithm")
        .agg(
            diversity_gain_vs_marco=("diversity_gain_vs_marco", "mean"),
            diversity_gain_vs_marco_topk=("diversity_gain_vs_marco_topk", "mean"),
            additional_runtime_vs_marco=("additional_runtime_vs_marco", "mean"),
        )
        .sort_values("diversity_gain_vs_marco_topk", ascending=False)
    )

    result.to_csv(output_csv, quoting=csv.QUOTE_MINIMAL)


def plots_1_instance(input_csv: Path) -> None:
    df = pd.read_csv(input_csv)
    div_matrices = []
    for _, row in df.iterrows():
        muses = parse_mus_list(str(row["MUSes"]))
        div_matrix = diversity_matrix(muses)
        div_matrices.append(div_matrix)
        print(f"Algorithm: {row['algorithm']}")
        visualize_heatmap(div_matrix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MUS result CSV.")
    parser.add_argument("input_csv", type=Path, help="Path to the original CSV file")
    parser.add_argument("output_dir", type=Path, help="Path to the output directory")
    args = parser.parse_args()

    diversity_per_instance(args.input_csv, args.output_dir/"diversity_per_instance.csv")
    topk_diversity_per_instance(args.output_dir/"xcsp_topk_20260412_141740.csv", args.output_dir/"topk_per_instance.csv")
    compare_to_topk_per_instance(args.output_dir/"diversity_per_instance.csv", args.output_dir/"topk_per_instance.csv", args.output_dir/"compare_to_topk_per_instance.csv")
    summarize_topk_comparison(args.output_dir/"compare_to_topk_per_instance.csv", args.output_dir/"summary_topk_comparison.csv")
    compare_to_marco_per_instance(args.output_dir/"diversity_per_instance.csv", args.output_dir/"compare_per_instance.csv")
    summarize_per_algorithm(args.output_dir/"compare_per_instance.csv", args.output_dir/"summary_per_algorithm.csv")
    summarize_completed(args.output_dir/"completed_instances.csv", args.output_dir/"compare_to_topk_per_instance.csv", args.output_dir/"summary_completed.csv")
    # plots_1_instance(args.input_csv)

if __name__ == "__main__":
    main()
