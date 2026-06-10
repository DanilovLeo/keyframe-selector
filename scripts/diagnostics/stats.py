"""
stats.py  (Tier-1 diagnostic)

Puts proper uncertainty on the headline retrieval grid so "method X beats Y"
claims can be tested instead of eyeballed.  With only 178 queries the binomial
95% CI on a Top-1 of ~0.82 is ~±0.056 — wider than the entire between-method
spread — so the expectation is that no pair is significant.  This script shows
that rather than asserting it.

Reconstructs each existing config's demo embeddings from the exported keyframe
indices, then:
  * bootstrap 95% CIs on Top-1 / Top-5 (resampling the 178 queries), and
  * paired permutation tests (per-query correctness, sign-flip null) for each
    CV / random method vs the uniform baseline at every K, plus the single
    largest pairwise gap in the grid.

Random is aggregated across its 3 seeds (per-query mean correctness) to match the
published mean±std reporting.

Outputs:
  results/tables/retrieval_cis.{md,csv}
  results/tables/retrieval_permutation.{md,csv}

Requires the bundle to include keyframes.jsonl (export without --no_indices).

Usage (from keyframe-selector/):
    python scripts/diagnostics/stats.py --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from bundle import Bundle, per_query_correct  # noqa: E402

K_SWEEP = [4, 8, 16, 32]
RANDOM_SEEDS = [42, 123, 456]
METHODS = ["uniform", "random", "optical_flow", "attention", "frame_diff"]


def per_query_correctness(b: Bundle, label: str) -> dict[int, np.ndarray]:
    """{k: per-query correctness} for one exported label, in query-eps order."""
    g_e, g_l, q_e, q_l = b.demo_embeddings_for_label(label)
    return per_query_correct(g_e, g_l, q_e, q_l, top_k=(1, 5))


def method_correctness(b: Bundle, method: str, k: int) -> dict[int, np.ndarray]:
    """Per-query correctness for (method, K); random is averaged over its seeds."""
    if method == "random":
        per_seed = [per_query_correctness(b, f"random_k{k}_s{s}") for s in RANDOM_SEEDS]
        return {kk: np.mean([d[kk].astype(float) for d in per_seed], axis=0)
                for kk in (1, 5)}
    d = per_query_correctness(b, f"{method}_k{k}")
    return {kk: d[kk].astype(float) for kk in (1, 5)}


def bootstrap_ci(correct: np.ndarray, b_iters: int, rng, alpha: float = 0.05):
    n = len(correct)
    idx = rng.integers(0, n, size=(b_iters, n))
    means = correct[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(correct.mean()), float(lo), float(hi)


def perm_test(cA: np.ndarray, cB: np.ndarray, b_iters: int, rng):
    """Two-sided paired permutation test (sign-flip null) on mean difference."""
    d = cA - cB
    obs = float(d.mean())
    signs = rng.choice([1.0, -1.0], size=(b_iters, len(d)))
    perm = (signs * d).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return obs, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    b = Bundle(args.bundle)
    if not b.has_indices():
        sys.exit("Bundle has no keyframes.jsonl — re-export without --no_indices.")
    rng = np.random.default_rng(args.seed)
    print(f"Loaded bundle: {len(b.query_eps())} queries, {len(b.gallery_eps())} gallery")

    # ---- per-(method, K) correctness, cached for reuse ------------------- #
    corr: dict[tuple, dict] = {}
    for m in METHODS:
        for k in K_SWEEP:
            corr[(m, k)] = method_correctness(b, m, k)

    # ---- bootstrap CIs --------------------------------------------------- #
    ci_rows = []
    for m in METHODS:
        for k in K_SWEEP:
            t1, lo1, hi1 = bootstrap_ci(corr[(m, k)][1], args.boot, rng)
            t5, lo5, hi5 = bootstrap_ci(corr[(m, k)][5], args.boot, rng)
            ci_rows.append({"method": m, "K": k,
                            "top_1": t1, "t1_lo": lo1, "t1_hi": hi1,
                            "top_5": t5, "t5_lo": lo5, "t5_hi": hi5})
    ci_df = pd.DataFrame(ci_rows)

    # ---- permutation tests: each method vs uniform, per K ---------------- #
    perm_rows = []
    for k in K_SWEEP:
        cu = corr[("uniform", k)][1]
        for m in METHODS:
            if m == "uniform":
                continue
            obs, p = perm_test(corr[(m, k)][1], cu, args.boot, rng)
            perm_rows.append({"K": k, "method_a": m, "method_b": "uniform",
                              "diff_top1": obs, "p_value": p})

    # ---- the single largest pairwise gap in the grid --------------------- #
    best = None
    for k in K_SWEEP:
        for i, ma in enumerate(METHODS):
            for mb in METHODS[i + 1:]:
                obs, p = perm_test(corr[(ma, k)][1], corr[(mb, k)][1], args.boot, rng)
                if best is None or abs(obs) > abs(best["diff_top1"]):
                    best = {"K": k, "method_a": ma, "method_b": mb,
                            "diff_top1": obs, "p_value": p}
    if best:
        best = {**best, "method_b": best["method_b"] + " (LARGEST GAP)"}
        perm_rows.append(best)
    perm_df = pd.DataFrame(perm_rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    ci_df.to_csv(out / "tables" / "retrieval_cis.csv", index=False)
    perm_df.to_csv(out / "tables" / "retrieval_permutation.csv", index=False)
    _write_md(ci_df, out / "tables" / "retrieval_cis.md")
    _write_md(perm_df, out / "tables" / "retrieval_permutation.md")

    print("\n--- retrieval_cis ---")
    print(ci_df.to_string(index=False))
    print("\n--- retrieval_permutation (vs uniform; last row = largest grid gap) ---")
    print(perm_df.to_string(index=False))
    n_sig = int((perm_df["p_value"] < 0.05).sum())
    print(f"\nSignificant comparisons at p<0.05: {n_sig} / {len(perm_df)}")


def _write_md(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    path.write_text("\n".join([head, sep, *body]) + "\n")


if __name__ == "__main__":
    main()
