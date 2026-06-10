"""
coverage_error.py  (Tier-2 diagnostic — fourth intrinsic metric, reported alongside retrieval)

A selection-sensitive intrinsic metric to contrast against the saturated
retrieval metric. See docs/decisions.md (2026-06-10, ADOPTED) for the scope
framing: this is an EVALUATION metric computed post-hoc on frozen, cached CLIP
embeddings — not a training loss, no reconstruction, no optimisation — so it
stays inside the pixels-only, CV-only Variant C scope. Reported alongside the
original three metrics, never as a standalone headline.

Definition. For an episode with per-frame L2-normalised CLIP embeddings
e_1..e_T and a selector's keyframe set S, every non-selected frame t is scored
by its cosine distance to the nearest selected frame:

    d(t) = min_{s in S} (1 - cos(e_t, e_s))

We report, per method × K, the episode mean of mean_t d(t) (typical distortion)
and the episode mean of max_t d(t) (worst uncovered frame). This is
vector-quantisation distortion with the keyframe set as the codebook. Lower is
better. The consecutive-block control (all K frames from the episode start) is
included as the key contrast: it should score far worse than evenly-spread
selection, even though mean-pooled retrieval cannot distinguish it.

Outputs:
  results/tables/coverage_error.{md,csv}

Usage (from keyframe-selector/):
    python scripts/diagnostics/coverage_error.py --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from bundle import Bundle  # noqa: E402

K_SWEEP = [4, 8, 16, 32]
RANDOM_SEEDS = [42, 123, 456]
METHODS = ["uniform", "random", "optical_flow", "attention", "frame_diff"]


def episode_coverage(E: np.ndarray, sel_idx) -> tuple[float, float]:
    """(mean, max) nearest-anchor cosine distance over non-selected frames.

    E is (T, D) L2-normalised; sel_idx is the keyframe index set. A frame that is
    itself selected has distance 0 and is excluded (per the audit's "for each
    non-selected frame"). If every frame is selected (T <= K), coverage is 0.
    """
    T = E.shape[0]
    sel = np.unique(np.asarray(sel_idx, dtype=int))
    mask = np.ones(T, dtype=bool)
    mask[sel] = False
    if not mask.any():
        return 0.0, 0.0
    sims = E[mask] @ E[sel].T          # (T-|S|, |S|) cosine, embeddings are unit-norm
    dmin = 1.0 - sims.max(axis=1)      # nearest-anchor cosine distance per non-sel frame
    return float(dmin.mean()), float(dmin.max())


def _label_indices(b: Bundle, label: str):
    return lambda ep, T: b.indices(ep, label)


def _consecutive_indices(k: int):
    return lambda ep, T: list(range(min(k, T)))


def coverage_for_rule(b: Bundle, idx_fn) -> tuple[float, float, int]:
    means, maxes = [], []
    for ep in b.episode_indices:
        E = b.frames(ep)
        m, x = episode_coverage(E, idx_fn(ep, E.shape[0]))
        means.append(m); maxes.append(x)
    return float(np.mean(means)), float(np.mean(maxes)), len(means)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    if not b.has_indices():
        sys.exit("Bundle has no keyframes.jsonl — re-export without --no_indices.")
    print(f"Loaded bundle: {len(b.episode_indices)} episodes")

    rows = []
    for k in K_SWEEP:
        for m in METHODS:
            if m == "random":
                ms, xs = [], []
                for s in RANDOM_SEEDS:
                    mm, xx, _ = coverage_for_rule(b, _label_indices(b, f"random_k{k}_s{s}"))
                    ms.append(mm); xs.append(xx)
                mean_cov, max_cov, n = float(np.mean(ms)), float(np.mean(xs)), len(b.episode_indices)
            else:
                mean_cov, max_cov, n = coverage_for_rule(b, _label_indices(b, f"{m}_k{k}"))
            rows.append({"method": m, "K": k, "mean_cov": mean_cov,
                         "max_cov": max_cov, "n_eps": n})
        # key contrast: worst-case temporal coverage at the same budget
        mc, xc, n = coverage_for_rule(b, _consecutive_indices(k))
        rows.append({"method": "consecutive_block", "K": k, "mean_cov": mc,
                     "max_cov": xc, "n_eps": n})

    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "coverage_error.csv", index=False)
    _write_md(df, out / "tables" / "coverage_error.md")

    print("\n" + df.to_string(index=False))
    print("\nwrote coverage_error.{md,csv}")


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
    banner = ("> Embedding-space coverage error (lower = better). Fourth intrinsic "
              "metric, reported alongside retrieval; see docs/decisions.md (2026-06-10).\n\n")
    path.write_text(banner + "\n".join([head, sep, *body]) + "\n")


if __name__ == "__main__":
    main()
