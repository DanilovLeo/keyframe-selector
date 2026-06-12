"""
instance_retrieval.py  (Task 5 — instance-level retrieval, within-episode half-split)

Re-frames the pinned retrieval metric with *episode identity* as the label
instead of the task label, to test whether selection matters at the harder
instance level even though it is invisible to task retrieval (§4-§5).
Pre-registered in docs/decisions.md (2026-06-12).

Protocol (pre-registered fork):
  - Split each episode's frames at the temporal midpoint mid = T//2:
    H1 = [0, mid), H2 = [mid, T).
  - Query vector  = pooled selected keyframes (full-episode exported indices)
                    that fall in H1.
  - Gallery entry = the same episode's selected keyframes that fall in H2.
  - Gallery = all episodes' H2 pools; queries = all episodes' H1 pools.
    A query is correct iff its nearest gallery vector is the SAME episode.
    Chance = 1 / N_episodes. Endpoints 0 and T-1 are forced -> halves non-empty.
  - Metric: Top-1 / Top-5 same-episode ID per method x K; paired sign-flip
    permutation test of per-query correctness vs uniform (random over seeds).

Outputs:
  results/tables/instance_retrieval.{md,csv}
  results/tables/instance_significance.{md,csv}

Pure numpy; reads only the bundle (needs exported keyframe indices). No GPU.

Usage (from keyframe-selector/):
    python scripts/diagnostics/instance_retrieval.py \\
        --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `bundle` import

import numpy as np
import pandas as pd

from bundle import Bundle, pool_demo_embedding, per_query_correct  # noqa: E402


def half_pools(E: np.ndarray, sel) -> tuple[np.ndarray, np.ndarray]:
    """(H1 pool, H2 pool) from the selected keyframe indices split at T//2."""
    T = E.shape[0]
    mid = T // 2
    sel = np.unique(np.asarray(sel, dtype=int))
    h1 = sel[sel < mid]
    h2 = sel[sel >= mid]
    # Endpoints are forced, so both are non-empty; guard anyway.
    if h1.size == 0:
        h1 = np.array([0])
    if h2.size == 0:
        h2 = np.array([T - 1])
    return pool_demo_embedding(E, h1), pool_demo_embedding(E, h2)


def build_QG(b: Bundle, label: str):
    """(Q, G, ep_ids): first-half query pools and second-half gallery pools."""
    Q, G, ids = [], [], []
    for ep in b.episode_indices:
        E = b.frames(ep)
        q, g = half_pools(E, b.indices(ep, label))
        Q.append(q); G.append(g); ids.append(ep)
    return np.stack(Q), np.stack(G), np.array(ids)


def correctness(b: Bundle, label: str, top_k=(1, 5)) -> dict:
    """{k: bool array over queries} for one label."""
    Q, G, ids = build_QG(b, label)
    return per_query_correct(G, ids, Q, ids, top_k=top_k)


def method_correct(b: Bundle, method: str, k: int, top_k=(1, 5)) -> dict:
    """Per-query correctness for a method at K (random averaged over seeds -> float)."""
    if method == "random":
        per_seed = [correctness(b, f"random_k{k}_s{s}", top_k) for s in b.random_seeds]
        return {kk: np.mean([ps[kk] for ps in per_seed], axis=0) for kk in top_k}
    return correctness(b, f"{method}_k{k}", top_k)


def perm_test(a: np.ndarray, base: np.ndarray, rng, iters: int = 10000):
    """Two-sided paired sign-flip permutation test on mean(a - base)."""
    d = a.astype(float) - base.astype(float)
    obs = float(d.mean())
    signs = rng.choice([1.0, -1.0], size=(iters, len(d)))
    perm = (signs * d).mean(axis=1)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return obs, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    if not b.has_indices():
        sys.exit("Bundle has no keyframes.jsonl — re-export without --no_indices.")
    n = len(b.episode_indices)
    print(f"Loaded bundle: {n} episodes  (instance chance = {1.0 / n:.4f})")

    top_k = (1, 5)
    # cache per-query correctness so the grid and the significance tests agree
    corr = {(m, k): method_correct(b, m, k, top_k) for k in b.k_sweep for m in b.methods}

    rows = []
    for k in b.k_sweep:
        for m in b.methods:
            c = corr[(m, k)]
            rows.append({"method": m, "K": k,
                         "top_1": float(np.mean(c[1])),
                         "top_5": float(np.mean(c[5])),
                         "n_queries": n, "chance": 1.0 / n})
    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "instance_retrieval.csv", index=False)
    _write_md(df, out / "tables" / "instance_retrieval.md",
              banner="> Instance-level retrieval: identify the same episode from its "
                     "other half (Top-1/Top-5). See docs/decisions.md (2026-06-12).")

    # ---- paired significance vs uniform (Top-1) -------------------------- #
    rng = np.random.default_rng(0)
    sig_rows = []
    n_sig = 0
    for k in b.k_sweep:
        uni = corr[("uniform", k)][1].astype(float)
        for m in b.methods:
            if m == "uniform":
                continue
            arr = corr[(m, k)][1].astype(float)
            obs, p = perm_test(arr, uni, rng)
            sig = p < 0.05
            n_sig += int(sig)
            sig_rows.append({"method": m, "vs": "uniform", "K": k,
                             "diff_top1": obs, "p_value": p, "sig_0.05": sig})
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(out / "tables" / "instance_significance.csv", index=False)
    _write_md(sig_df, out / "tables" / "instance_significance.md",
              banner="> Paired sign-flip permutation tests on per-query Top-1 instance "
                     "correctness (diff = method - uniform). 863 paired queries.")

    print("\n" + df.to_string(index=False))
    print("\n--- paired significance vs uniform (Top-1) ---")
    print(sig_df.to_string(index=False))
    print(f"\nVERDICT: {n_sig} of {len(sig_rows)} method-vs-uniform Top-1 comparisons "
          f"significant at p<0.05  "
          f"=> {'SELECTION-SENSITIVE' if n_sig > 0 else 'SELECTION-INVARIANT'} at instance level")
    print("\nwrote instance_retrieval.{md,csv} and instance_significance.{md,csv}")


def _write_md(df: pd.DataFrame, path: Path, banner: str) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    path.write_text(banner + "\n\n" + "\n".join([head, sep, *body]) + "\n")


if __name__ == "__main__":
    main()
