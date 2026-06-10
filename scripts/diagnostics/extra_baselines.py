"""
extra_baselines.py  (Tier-1 diagnostic)

Adds the missing controls a reviewer asks for first, all reconstructed from the
exported embeddings (no GPU, no re-run):

  * K=1 single-frame baselines (frame 0, middle frame) — if retrieval barely
    drops at K=1, "one frame suffices for scene recognition" is the cleanest
    demonstration of saturation.
  * consecutive-block control — K consecutive frames from the episode start, i.e.
    worst-case temporal coverage.  If a degenerate block still scores ~0.82, the
    metric is insensitive even to maximally pathological selection (stronger than
    random parity).
  * oracle upper bound — query-side selection that uses the label to maximise
    retrieval.  If the ceiling is ~0.85, no selection method has more than a
    couple points of headroom, so the negative result is a property of the
    benchmark, not a failure of the methods.
        - oracle_k1_exists : exact K=1 ceiling (does ANY single frame retrieve
                             the correct task at top-1?).
        - oracle_k4_greedy : greedy margin maximisation at K=4 (approximate
                             ceiling; exhaustive C(T,4) search is intractable).

These intentionally break the "always include frame 0 and T-1" convention — they
are probes of the metric, not proposed extractors.  The oracle uses the label
for SELECTION only, as an upper bound; it is never a deployable method.

Outputs:
  results/tables/extra_baselines.{md,csv}

Usage (from keyframe-selector/):
    python scripts/diagnostics/extra_baselines.py --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from bundle import Bundle, pool_demo_embedding, retrieval_accuracy  # noqa: E402


# --------------------------------------------------------------------------- #
# Simple, fixed selection rules (gallery and query both use the rule)
# --------------------------------------------------------------------------- #
def sel_frame0(ep, T, E):
    return [0]


def sel_mid(ep, T, E):
    return [T // 2]


def make_consecutive(k):
    def sel(ep, T, E):
        n = min(k, T)
        return list(range(n))            # block from the start = worst coverage
    return sel


def cr_stats(bundle: Bundle, select_fn):
    crs, nkf = [], []
    for ep in bundle.episode_indices:
        T = bundle.ep_T[ep]
        n = len(np.asarray(select_fn(ep, T, bundle.frames(ep))))
        crs.append(n / T if T else 0.0)
        nkf.append(n)
    return float(np.mean(crs)), float(np.mean(nkf))


def eval_fixed_rule(bundle: Bundle, name: str, select_fn) -> dict:
    g_e, g_l, q_e, q_l = bundle.demo_embeddings(select_fn)
    acc = retrieval_accuracy(g_e, g_l, q_e, q_l, top_k=(1, 5))
    mcr, mkf = cr_stats(bundle, select_fn)
    return {"baseline": name, **acc, "mean_cr": mcr, "mean_n_kf": mkf,
            "n_gallery": len(g_l), "n_query": len(q_l)}


# --------------------------------------------------------------------------- #
# Oracle upper bounds (query-side selection; gallery = full-episode mean)
# --------------------------------------------------------------------------- #
def oracle_bounds(bundle: Bundle) -> list[dict]:
    g_eps = bundle.gallery_eps()
    g_embs = np.stack([bundle.episode_mean(ep) for ep in g_eps])   # (Ng, D)
    g_labels = np.array([bundle.ep_task[ep] for ep in g_eps])
    q_eps = bundle.query_eps()

    k1_exists = np.zeros(len(q_eps), dtype=bool)
    k4_greedy = np.zeros(len(q_eps), dtype=bool)

    for i, ep in enumerate(q_eps):
        E = bundle.frames(ep)                       # (T, D) L2-normalised
        t = bundle.ep_task[ep]
        same = g_labels == t
        if not same.any() or same.all():
            # degenerate: no contrastive gallery; treat as trivially correct
            k1_exists[i] = True
            k4_greedy[i] = True
            continue

        # --- exact K=1 ceiling: does any single frame retrieve task t? ----- #
        sim_f = E @ g_embs.T                         # (T, Ng) per-frame sims
        best_g = np.argmax(sim_f, axis=1)            # (T,)
        k1_exists[i] = bool((g_labels[best_g] == t).any())

        # --- greedy K=4 by retrieval margin -------------------------------- #
        chosen, ssum = [], np.zeros(E.shape[1], dtype=np.float64)
        for _ in range(min(4, E.shape[0])):
            best_margin, best_f = -np.inf, -1
            for f in range(E.shape[0]):
                if f in chosen:
                    continue
                pooled = (ssum + E[f]) / (len(chosen) + 1)
                pooled = pooled / max(np.linalg.norm(pooled), 1e-8)
                sims = pooled @ g_embs.T
                margin = sims[same].max() - sims[~same].max()
                if margin > best_margin:
                    best_margin, best_f = margin, f
            chosen.append(best_f)
            ssum += E[best_f]
        dem = ssum / len(chosen)
        dem = dem / max(np.linalg.norm(dem), 1e-8)
        sims = dem @ g_embs.T
        k4_greedy[i] = bool(g_labels[np.argmax(sims)] == t)

    return [
        {"baseline": "oracle_k1_exists", "top_1": float(k1_exists.mean()),
         "top_5": float("nan"), "mean_cr": float("nan"), "mean_n_kf": 1.0,
         "n_gallery": len(g_labels), "n_query": len(q_eps)},
        {"baseline": "oracle_k4_greedy", "top_1": float(k4_greedy.mean()),
         "top_5": float("nan"), "mean_cr": float("nan"), "mean_n_kf": 4.0,
         "n_gallery": len(g_labels), "n_query": len(q_eps)},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    print(f"Loaded bundle: {len(b.episodes)} episodes")

    rows = [
        eval_fixed_rule(b, "frame0_k1", sel_frame0),
        eval_fixed_rule(b, "mid_k1", sel_mid),
        eval_fixed_rule(b, "consecutive_k4_start", make_consecutive(4)),
        eval_fixed_rule(b, "consecutive_k8_start", make_consecutive(8)),
    ]
    rows.extend(oracle_bounds(b))
    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "extra_baselines.csv", index=False)
    _write_md(df, out / "tables" / "extra_baselines.md")

    print("\n" + df.to_string(index=False))
    print("\nwrote extra_baselines.{md,csv}")


def _write_md(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    path.write_text("\n".join([head, sep, *body]) + "\n")


if __name__ == "__main__":
    main()
