"""
pooling_sensitivity.py  (Tier-3 diagnostic)

Sensitivity analysis on the pinned aggregation step, NOT a protocol change. The
headline retrieval protocol mean-pools L2-normalised CLIP keyframe embeddings;
mean-pooling is order-invariant and near-idempotent on a tight cluster, which is
exactly why it cannot resolve *which* frames are kept (methods.md §4.1). This
script re-runs the full method × K retrieval grid under two alternative
aggregators that do NOT collapse the keyframe set into one averaged point, and
asks a single question: does any of them make the methods separable?

  * mean        — the pinned protocol, recomputed here as a sanity reference
                  (must reproduce results/tables/retrieval_top1_pivot.md).
  * max         — element-wise max over the selected frame embeddings, then
                  L2-normalise. A coordinate-wise extremum keeps per-frame
                  outliers instead of averaging them away.
  * best_match  — no pooling at all: each demo is its *set* of keyframe
                  embeddings; similarity(query, gallery) = max cosine over all
                  query-frame × gallery-frame pairs (best-matching frame pair).
                  Selection-sensitive by construction.

Random is averaged over its 3 seeds to match the published reporting. The metric
of interest is the between-method spread at each K: if it stays inside the n=178
binomial noise band (~±0.056) under every aggregator, scene-dominance is
reinforced; if max/best_match opens a real gap, that is a second mechanism
result.

Outputs:
  results/tables/pooling_sensitivity.{md,csv}

Usage (from keyframe-selector/):
    python scripts/diagnostics/pooling_sensitivity.py --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from bundle import Bundle, per_query_correct  # noqa: E402

# The experiment grid (K-sweep, seeds, methods) is read from the bundle metadata
# at runtime — see Bundle.k_sweep / .random_seeds / .methods — so this script
# stays in lock-step with the grid the bundle was exported with. The single
# human-editable source for that grid is configs/experiment.yaml.
TOP_K = (1, 5)


def _norm(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-8)


def build_sets(b: Bundle, label: str):
    """(gallery, query) lists of (task_id, selected_frames) using exported indices."""
    g, q = [], []
    for ep in b.episode_indices:
        E = b.frames(ep)
        sel = E[np.asarray(b.indices(ep, label), dtype=int)]
        rec = (b.ep_task[ep], sel)
        (g if b.ep_split[ep] == "gallery" else q).append(rec)
    return g, q


# --------------------------------------------------------------------------- #
# Vector aggregators (one (D,) vector per demo, then cosine retrieval)
# --------------------------------------------------------------------------- #
def _pooled_vectors(sets, pool):
    labels = np.array([t for t, _ in sets])
    vecs = np.stack([_norm(pool(sel)) for _, sel in sets])
    return vecs, labels


def _vector_correct(b: Bundle, label: str, pool) -> dict[int, np.ndarray]:
    g, q = build_sets(b, label)
    g_e, g_l = _pooled_vectors(g, pool)
    q_e, q_l = _pooled_vectors(q, pool)
    return per_query_correct(g_e, g_l, q_e, q_l, top_k=TOP_K)


def mean_pool(sel):
    return sel.mean(axis=0)


def max_pool(sel):
    return sel.max(axis=0)


# --------------------------------------------------------------------------- #
# best_match: set-vs-set max-cosine retrieval (no pooling)
# --------------------------------------------------------------------------- #
def _best_match_correct(b: Bundle, label: str) -> dict[int, np.ndarray]:
    g, q = build_sets(b, label)
    g_l = np.array([t for t, _ in g])
    g_frames = [sel for _, sel in g]
    G = np.concatenate(g_frames, axis=0)                       # (sum_Kg, D)
    sizes = [len(s) for s in g_frames]
    starts = np.r_[0, np.cumsum(sizes)[:-1]]                   # segment starts in G

    per = {k: np.zeros(len(q), dtype=bool) for k in TOP_K}
    for i, (t, sel) in enumerate(q):
        sim = sel @ G.T                                       # (Kq, sum_Kg)
        sim = sim.max(axis=0)                                 # best query frame per gallery frame
        scores = np.maximum.reduceat(sim, starts)            # (Ng,) per-gallery-demo max
        ranked = np.argsort(-scores)
        for k in TOP_K:
            keff = min(k, len(g_l))
            per[k][i] = bool((g_l[ranked[:keff]] == t).any())
    return per


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def _method_correct(b: Bundle, pooling: str, method: str, k: int) -> dict[int, np.ndarray]:
    def one(label):
        if pooling == "mean":
            return _vector_correct(b, label, mean_pool)
        if pooling == "max":
            return _vector_correct(b, label, max_pool)
        if pooling == "best_match":
            return _best_match_correct(b, label)
        raise ValueError(pooling)

    if method == "random":
        per_seed = [one(f"random_k{k}_s{s}") for s in b.random_seeds]
        return {kk: np.mean([d[kk].astype(float) for d in per_seed], axis=0) for kk in TOP_K}
    d = one(f"{method}_k{k}")
    return {kk: d[kk].astype(float) for kk in TOP_K}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    if not b.has_indices():
        sys.exit("Bundle has no keyframes.jsonl — re-export without --no_indices.")
    print(f"Loaded bundle: {len(b.query_eps())} queries, {len(b.gallery_eps())} gallery")

    rows, spread_rows = [], []
    for pooling in ["mean", "max", "best_match"]:
        for k in b.k_sweep:
            t1_by_method = {}
            for m in b.methods:
                c = _method_correct(b, pooling, m, k)
                t1, t5 = float(c[1].mean()), float(c[5].mean())
                t1_by_method[m] = t1
                rows.append({"pooling": pooling, "method": m, "K": k,
                             "top_1": t1, "top_5": t5})
            vals = np.array(list(t1_by_method.values()))
            spread_rows.append({"pooling": pooling, "K": k,
                                "top1_min": float(vals.min()),
                                "top1_max": float(vals.max()),
                                "top1_spread": float(vals.max() - vals.min()),
                                "argmax_method": max(t1_by_method, key=t1_by_method.get)})
        print(f"  done pooling={pooling}")

    df = pd.DataFrame(rows)
    spread = pd.DataFrame(spread_rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "pooling_sensitivity.csv", index=False)
    _write_md(df, spread, out / "tables" / "pooling_sensitivity.md")

    print("\n--- per (pooling, method, K) ---")
    print(df.to_string(index=False))
    print("\n--- between-method Top-1 spread per (pooling, K) ---")
    print(spread.to_string(index=False))
    print("\nwrote pooling_sensitivity.{md,csv}")


def _fmt(v):
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def _df_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _write_md(df: pd.DataFrame, spread: pd.DataFrame, path: Path) -> None:
    text = ("## Pooling sensitivity — per (pooling, method, K)\n\n"
            + _df_md(df)
            + "\n\n## Between-method Top-1 spread per (pooling, K)\n\n"
            + _df_md(spread) + "\n")
    path.write_text(text)


if __name__ == "__main__":
    main()
