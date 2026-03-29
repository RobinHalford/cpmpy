from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import List, Sequence, Set
import pandas as pd

from cpmpy.tools.explain.utils import average_diversity 

TOKEN_RE = re.compile(r"[^,\s\[\]\{\}]+")


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


def parse_mus_list(text: str) -> List[Set[str]]:
    """
    Parse a MUS collection string like:
        "[{BV1, BV2}, {BV3, BV4}]"
        "[[BV1, BV2], [BV3, BV4]]"
    into:
        [{"BV1", "BV2"}, {"BV3", "BV4"}]
    """
    text = (text or "").strip()
    if not text or text == "[]":
        return []

    muses: List[Set[str]] = []
    current: List[str] = []
    depth = 0
    in_mus = False

    for ch in text:
        if ch in "{[":
            depth += 1
            # depth 2 means we just entered one MUS inside the outer list
            if depth == 2:
                in_mus = True
                current = []
        elif ch in "}]":
            if depth == 2 and in_mus:
                tokens = TOKEN_RE.findall("".join(current))
                mus = {tok for tok in tokens if tok}
                if mus:
                    muses.append(mus)
                in_mus = False
                current = []
            depth -= 1
        else:
            if in_mus:
                current.append(ch)

    return muses


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MUS result CSV.")
    parser.add_argument("input_csv", type=Path, help="Path to the original CSV file")
    parser.add_argument("output_csv", type=Path, help="Path to the summary CSV file")
    args = parser.parse_args()

    summarize(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
