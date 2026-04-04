from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path
from typing import List, Sequence
import pandas as pd
import numpy as np

from cpmpy.tools.explain.utils import average_diversity, diversity_matrix
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


def summarize(input_csv: Path, output_csv: Path) -> None:
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
                "avg_diversity": average_diversity(muses) if len(muses) > 1 else 0,
            }
        )

    summary_df = pd.DataFrame(rows)

    summary_df.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)


def analyze_summary(summary_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(summary_csv)

    required_cols = {
        "instance",
        "algorithm",
        "status",
        "total_runtime",
        "avg_diversity",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    # Start with NaN for everything
    df["diversity_gain_vs_marco"] = np.nan
    df["runtime_increase_pct_vs_marco"] = np.nan

    marco_mask = df["algorithm"].eq("marco")
    df.loc[marco_mask, ["diversity_gain_vs_marco", "runtime_increase_pct_vs_marco"]] = 0.0

    marco_baseline = (
        df.loc[marco_mask & df["status"].eq("COMPLETE"), ["instance", "avg_diversity", "total_runtime"]]
        .rename(
            columns={
                "avg_diversity": "marco_avg_diversity",
                "total_runtime": "marco_total_runtime",
            }
        )
    )

    df = df.merge(marco_baseline, on="instance", how="left")
    completed_non_marco = df["status"].eq("COMPLETE") & ~df["algorithm"].eq("marco")

    df.loc[completed_non_marco, "diversity_gain_vs_marco"] = (
        df.loc[completed_non_marco, "avg_diversity"]
        - df.loc[completed_non_marco, "marco_avg_diversity"]
    )

    df.loc[completed_non_marco, "runtime_increase_pct_vs_marco"] = (
        (
            df.loc[completed_non_marco, "total_runtime"]
            - df.loc[completed_non_marco, "marco_total_runtime"]
        )
        / df.loc[completed_non_marco, "marco_total_runtime"]
        * 100
    )

    df = df.drop(columns=["marco_avg_diversity", "marco_total_runtime"])

    df.to_csv(output_csv, index=False)


def summarize_per_algorithm(compare_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(compare_csv)

    required_cols = {
        "algorithm",
        "status",
        "diversity_gain_vs_marco",
        "runtime_increase_pct_vs_marco",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    
    df = df[df["algorithm"] != "marco"].copy()
    is_timeout = df["status"].eq("TIMEOUT")
    is_completed = df["status"].eq("COMPLETE")
    timeout_pct = (df.groupby("algorithm")["status"].apply(lambda s: 100 * s.eq("TIMEOUT").mean()).rename("timeout_pct"))

    completed_instances = df.groupby("instance")["status"].apply(lambda s: s.eq("COMPLETE").all())

    df_all_completed = df[df["instance"].isin(completed_instances[completed_instances].index)]
    # df_completed = df[is_completed]
    df_all_completed.to_csv(output_csv.parent / "completed_instances.csv", index=False)

    agg = (
        df_all_completed.groupby("algorithm")
        .agg(
            avg_diversity_gain_vs_marco=("diversity_gain_vs_marco", "mean"),
            avg_runtime_change_pct_vs_marco=("runtime_increase_pct_vs_marco", "mean"),
        )
    )
    result = timeout_pct.to_frame().join(agg)
    result.to_csv(output_csv)


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

    # summarize(args.input_csv, args.output_dir/"diversity.csv")
    # analyze_summary(args.output_dir/"diversity.csv", args.output_dir/"compare.csv")
    # summarize_per_algorithm(args.output_dir/"compare.csv", args.output_dir/"summary_all_completed.csv")
    plots_1_instance(args.input_csv)

if __name__ == "__main__":
    main()
