"""
similarity_distributions.py  (Tier-1 diagnostic)

Proves the saturation *mechanism* quantitatively instead of narrating it.

The claim in methods.md §4.1 is that BridgeData v2 task identity is
scene-dominated, so any subset mean of an episode's frame embeddings lands in a
tight cone and retrieval cannot resolve *which* frames were kept.  This script
turns that into a figure by comparing three cosine-similarity distributions on
the exported CLIP embeddings:

  1. intra-episode      cos(frame_i, frame_j) within the same episode
  2. inter-episode/task cos(mean_e, mean_e') for two episodes of the SAME task
  3. inter-task         cos(mean_e, mean_e') for episodes of DIFFERENT tasks

If the intra-episode distribution sits high and tight (frames already nearly
identical) and the same-task/different-task means are well separated, then
moving the keyframe subset around can only slide you within the intra-episode
cone — far smaller than the task gap — so selection cannot change retrieval.
That is the saturation result, shown rather than asserted.

Outputs:
  results/plots/fig5_similarity_distributions.{pdf,png}
  results/tables/similarity_distributions.{md,csv}

Pure numpy + matplotlib; reads only the bundle.

Usage (from keyframe-selector/):
    python scripts/diagnostics/similarity_distributions.py \\
        --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `bundle` import

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bundle import Bundle  # noqa: E402


def intra_episode_cosines(bundle: Bundle, max_pairs_per_ep: int = 2000,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Pooled distribution of within-episode pairwise frame cosine similarities."""
    rng = rng or np.random.default_rng(0)
    vals = []
    for ep in bundle.episode_indices:
        E = bundle.frames(ep)                      # (T, D), already L2-normalised
        T = E.shape[0]
        if T < 2:
            continue
        S = E @ E.T                                # (T, T)
        iu = np.triu_indices(T, k=1)               # upper triangle, no diagonal
        v = S[iu]
        if v.size > max_pairs_per_ep:              # bound memory on long episodes
            v = rng.choice(v, size=max_pairs_per_ep, replace=False)
        vals.append(v)
    return np.concatenate(vals) if vals else np.array([])


def mean_pair_cosines(bundle: Bundle) -> tuple[np.ndarray, np.ndarray]:
    """cos between episode-mean embeddings, split into same-task / diff-task."""
    eps = bundle.episode_indices
    M = np.stack([bundle.episode_mean(ep) for ep in eps])   # (N, D)
    tasks = np.array([bundle.ep_task[ep] for ep in eps])
    S = M @ M.T                                             # (N, N)
    iu = np.triu_indices(len(eps), k=1)
    same = tasks[iu[0]] == tasks[iu[1]]
    cos = S[iu]
    return cos[same], cos[~same]


def summary_row(name: str, v: np.ndarray) -> dict:
    q = np.percentile(v, [5, 25, 50, 75, 95]) if v.size else [np.nan] * 5
    return {
        "distribution": name,
        "n": int(v.size),
        "mean": float(np.mean(v)) if v.size else float("nan"),
        "std": float(np.std(v)) if v.size else float("nan"),
        "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    print(f"Loaded bundle: {len(b.episodes)} episodes, "
          f"{len(set(b.ep_task.values()))} tasks")

    intra = intra_episode_cosines(b)
    same_task, diff_task = mean_pair_cosines(b)

    rows = [
        summary_row("intra_episode_frame_pairs", intra),
        summary_row("inter_episode_same_task_means", same_task),
        summary_row("inter_task_means", diff_task),
    ]
    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "tables" / "similarity_distributions.csv", index=False)
    _write_md(df, out / "tables" / "similarity_distributions.md")

    # ---- figure ---------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.linspace(0.0, 1.0, 80)
    for name, v, color in [
        ("intra-episode (frame pairs)", intra, "#1f77b4"),
        ("inter-episode, same task (means)", same_task, "#2ca02c"),
        ("inter-task (means)", diff_task, "#d62728"),
    ]:
        if v.size:
            ax.hist(v, bins=bins, density=True, alpha=0.55, label=name, color=color)
            ax.axvline(np.median(v), color=color, lw=1.5, ls="--")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("CLIP embedding similarity: within-episode vs across episodes\n"
                 "(intra-episode tightness >> task gap  =>  selection-invariant retrieval)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / "plots" / f"fig5_similarity_distributions.{ext}", dpi=150)
    plt.close(fig)

    print("\n" + df.to_string(index=False))
    print(f"\nwrote fig5_similarity_distributions.* and similarity_distributions.*")

    # One-line interpretation aid.
    if intra.size and same_task.size and diff_task.size:
        print(f"\nintra-episode median = {np.median(intra):.3f}   "
              f"same-task means median = {np.median(same_task):.3f}   "
              f"inter-task means median = {np.median(diff_task):.3f}")


def _write_md(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    path.write_text("\n".join([head, sep, *body]) + "\n")


if __name__ == "__main__":
    main()
