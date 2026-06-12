"""
equivalence.py  (Task 2 — TOST equivalence bound on the retrieval grid)

§5.4 shows no *difference* between methods (0/40 permutation pairs significant).
That is absence of evidence, not evidence of absence. This script runs a paired
two-one-sided-test (TOST) on every method-pair Top-1 difference to ask the
positive question: are the methods *equivalent* within a pre-set margin δ?
Pre-registered in docs/decisions.md (2026-06-12), δ = 0.02, 90% CI.

For each pair (A, B) at each K, on paired per-query Top-1 correctness
(d_i = correct_A,i - correct_B,i; random averaged over its seeds):
  Δ          = mean(d)
  90% CI     = Δ ± t_{0.95, n-1} * se,   se = std(d, ddof=1) / sqrt(n)
  p_lower    tests H0: Δ <= -δ   (reject => Δ > -δ)
  p_upper    tests H0: Δ >= +δ   (reject => Δ < +δ)
  p_TOST     = max(p_lower, p_upper);  equivalent at α=0.05 iff 90% CI ⊂ (-δ, +δ)

Output:
  results/tables/equivalence_tost.{md,csv}

Pure numpy + scipy; reuses stats.method_correctness (cached embeddings only).
No GPU.  Re-runs unchanged on the 100-task bundle via --bundle.

Usage (from keyframe-selector/):
    python scripts/diagnostics/equivalence.py --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats as sstats

from bundle import Bundle           # noqa: E402
from stats import method_correctness  # noqa: E402  (exact §5.4 per-query correctness)


def tost_pair(cA: np.ndarray, cB: np.ndarray, delta: float, alpha: float = 0.05):
    """Paired TOST on mean(cA - cB). Returns dict of stats."""
    d = cA.astype(float) - cB.astype(float)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    df = n - 1
    tcrit = float(sstats.t.ppf(1 - alpha, df))   # one-sided crit => 90% two-sided CI
    ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
    if se > 0:
        t_lower = (mean + delta) / se            # H0: Δ <= -δ
        t_upper = (mean - delta) / se            # H0: Δ >= +δ
        p_lower = float(sstats.t.sf(t_lower, df))   # P(T >  t_lower)
        p_upper = float(sstats.t.cdf(t_upper, df))  # P(T <  t_upper)
    else:  # zero variance (identical) -> Δ=0 is inside (-δ,δ) iff δ>0
        p_lower = 0.0 if mean > -delta else 1.0
        p_upper = 0.0 if mean < delta else 1.0
    p_tost = max(p_lower, p_upper)
    equivalent = (ci_lo > -delta) and (ci_hi < delta)
    return {"diff_top1": mean, "se": se, "ci90_lo": ci_lo, "ci90_hi": ci_hi,
            "p_tost": p_tost, "equivalent": equivalent}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--delta", type=float, default=0.02)
    args = ap.parse_args()

    b = Bundle(args.bundle)
    if not b.has_indices():
        sys.exit("Bundle has no keyframes.jsonl — re-export without --no_indices.")
    methods, k_sweep = b.methods, b.k_sweep
    n_q = len(b.query_eps())
    print(f"Loaded bundle: {n_q} queries  (delta={args.delta})")

    corr = {(m, k): method_correctness(b, m, k) for m in methods for k in k_sweep}

    rows = []
    for k in k_sweep:
        for i, ma in enumerate(methods):
            for mb in methods[i + 1:]:
                r = tost_pair(corr[(ma, k)][1], corr[(mb, k)][1], args.delta)
                rows.append({"K": k, "method_a": ma, "method_b": mb, **r})
    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "equivalence_tost.csv", index=False)

    n_eq = int(df["equivalent"].sum())
    max_hw = float((df["ci90_hi"] - df["ci90_lo"]).max() / 2)
    median_hw = float(((df["ci90_hi"] - df["ci90_lo"]) / 2).median())
    note = [
        f"TOST equivalence at delta={args.delta}, 90% CI, n_queries={n_q} "
        "(docs/decisions.md 2026-06-12):",
        f"  pairs equivalent within +/-{args.delta}: {n_eq} / {len(df)}",
        f"  90% CI half-width: median {median_hw:.4f}, max {max_hw:.4f}",
        ("  => sample certifies +/-%.2f equivalence" % args.delta) if n_eq == len(df)
        else ("  => UNDERPOWERED for +/-%.2f: achievable bound ~+/-%.3f "
              "(needs more queries)" % (args.delta, max_hw)),
    ]
    _write_md(df, out / "tables" / "equivalence_tost.md", note=note)

    print("\n" + df.to_string(index=False))
    print("\n" + "\n".join(note))
    print("\nwrote equivalence_tost.{md,csv}")


def _write_md(df: pd.DataFrame, path: Path, note: list[str] | None = None) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    banner = ("> Paired TOST equivalence on Top-1 (diff = method_a - method_b). "
              "Equivalent = 90% CI within +/-delta. See docs/decisions.md "
              "(2026-06-12).\n\n")
    lines = [banner + head, sep, *body]
    if note:
        lines += ["", "```", *note, "```"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
