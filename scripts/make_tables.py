"""
make_tables.py

Write the result tables for the BridgeData v2 keyframe-selection study as both
Markdown and CSV (per CLAUDE.md: tables go in results/tables/ as both formats).
Companion to plot_results.py, which handles the figures.

Inputs (read-only):
  results/eval_retrieval.json           (run_retrieval_eval.py)
  results/consistency_check_bridge.json (run_consistency.py)

Outputs:
  results/tables/retrieval_summary.{md,csv}    one row per (method, K)
  results/tables/retrieval_top1_pivot.{md,csv} method x K grid of Top-1
  results/tables/consistency_aggregate.{md,csv} per-extractor mean across tasks

Pure JSON -> table; no torch, no GPU, runs in a second.

Usage (from keyframe-selector/):
    python scripts/make_tables.py [--results_dir results]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Anchored at end so 'random_k4_s42' (trailing _s42) does NOT match — we keep the
# aggregated 'random_k4' (mean ± std) and drop the per-seed entries. Mirrors the
# parser in plot_results.py.
_KEY_RE = re.compile(r"^(?P<method>.+)_k(?P<k>\d+)$")

# Display order for methods and consistency extractors.
METHOD_ORDER = ["uniform", "random", "optical_flow", "attention", "frame_diff"]
CONSISTENCY_ORDER = [
    "uniform_10", "random_10", "optical_flow", "attention_dino", "frame_diff",
]


def parse_method_k(key: str):
    m = _KEY_RE.match(key)
    return (m.group("method"), int(m.group("k"))) if m else None


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------ #
#  Markdown writer (no tabulate dependency)
# ------------------------------------------------------------------ #
def df_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body]) + "\n"


def write_table(df: pd.DataFrame, base: Path, floatfmt: str = "{:.3f}") -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base.with_suffix(".csv"), index=False)
    base.with_suffix(".md").write_text(df_to_markdown(df, floatfmt))
    print(f"  wrote {base.name}.md / .csv")


# ------------------------------------------------------------------ #
#  Table builders
# ------------------------------------------------------------------ #
def _method_rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def retrieval_summary(eval_retrieval: dict) -> pd.DataFrame:
    rows = []
    for key, e in eval_retrieval["results"].items():
        parsed = parse_method_k(key)
        if parsed is None:
            continue
        method, k = parsed
        rows.append({
            "method":    method,
            "K":         k,
            "top_1":     e["top_1"],
            "top_1_std": e.get("top_1_std", 0.0),
            "top_5":     e["top_5"],
            "top_5_std": e.get("top_5_std", 0.0),
            "clip_sim":  e["clip_sim"],
            "mean_cr":   e["mean_cr"],
            "mean_n_kf": e["mean_n_kf"],
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["method", "K"], key=lambda s: s.map(_method_rank)
                        if s.name == "method" else s)
    return df.reset_index(drop=True)


def retrieval_top1_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    piv = summary.pivot(index="method", columns="K", values="top_1")
    piv = piv.reindex([m for m in METHOD_ORDER if m in piv.index])
    piv.columns = [f"K={k}" for k in piv.columns]
    return piv.reset_index()


def consistency_aggregate(consistency: dict) -> pd.DataFrame:
    rows = []
    for ext in CONSISTENCY_ORDER:
        mean_kfs, cvs, crs = [], [], []
        for task_data in consistency.values():
            s = task_data["extractors"].get(ext)
            if s is None:
                continue
            mean_kfs.append(s["mean_kf"])
            cvs.append(s["cv_kf"])
            crs.append(s["mean_cr"])
        if not mean_kfs:
            continue
        rows.append({
            "extractor":   ext,
            "fixed_k":     ext in ("uniform_10", "random_10"),
            "n_tasks":     len(mean_kfs),
            "mean_kf":     float(np.mean(mean_kfs)),
            "cv_kf":       float(np.mean(cvs)),
            "mean_cr":     float(np.mean(crs)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    tdir = rdir / "tables"

    eval_retrieval = load_json(rdir / "eval_retrieval.json")
    consistency = load_json(rdir / "consistency_check_bridge.json")

    summary = retrieval_summary(eval_retrieval)
    pivot = retrieval_top1_pivot(summary)
    aggregate = consistency_aggregate(consistency)

    print("Writing tables:")
    write_table(summary,   tdir / "retrieval_summary")
    write_table(pivot,     tdir / "retrieval_top1_pivot")
    write_table(aggregate, tdir / "consistency_aggregate")

    print("\n--- retrieval_top1_pivot ---")
    print(pivot.to_string(index=False))
    print("\n--- consistency_aggregate ---")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
