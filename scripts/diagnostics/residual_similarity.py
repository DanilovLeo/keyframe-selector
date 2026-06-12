"""
residual_similarity.py  (Task 1, Stage 1 — pre-registered gate)

Repeats the §5.1 similarity-distribution analysis in *residual* embedding space
to test whether subtracting a per-episode scene anchor de-saturates the
representation enough for keyframe selection to matter.  This is the go/no-go
gate for the residual-retrieval work (Stage 2): see the pre-registered decision
rule in docs/decisions.md (entry 2026-06-12).

Residual variants (per-frame "direction of change", L2-normalised):
  raw: r_t = e_t                         (reproduces §5.1 exactly — baseline row)
  A:   r_t = normalize(e_t - e_0)        anchor = first frame
  B:   r_t = normalize(e_t - mean_t e_t) anchor = episode mean

For each variant we compute the same three cosine distributions as §5.1:
  1. intra-episode      cos(r_i, r_j) within an episode
  2. inter-episode/task cos(rmean_e, rmean_e') for episodes of the SAME task
  3. inter-task         cos(rmean_e, rmean_e') for episodes of DIFFERENT tasks
where the *episode-mean residual* rmean_e = normalize(mean_t (e_t - anchor)) is
the episode's net displacement from its anchor.

Degeneracy handling (documented; matches the 2026-06-12 pre-registration):
  - Frame 0 in variant A has r_0 = e_0 - e_0 = 0; excluded from intra stats.
  - Any per-frame residual with norm < EPS is set to the zero vector and dropped
    from the intra pool.
  - Variant B's episode-mean residual is identically 0 by construction
    (mean_t (e_t - mean(e)) = 0), so B's inter-episode / inter-task panels are
    UNDEFINED and reported empty; B contributes only its intra-episode row.

Pre-registered gate (evaluated on variant A):
  PASS -> run Stage 2   if median(intra_A) <= median(inter_task_A) - 0.05
                        and median(same_task_A) - median(inter_task_A) >= 0.02
  FAIL -> stop          otherwise.

Outputs:
  results/tables/residual_similarity.{md,csv}
  results/plots/fig_residual_similarity.{pdf,png}

Pure numpy + matplotlib; reads only the bundle.  No GPU.

Usage (from keyframe-selector/):
    python scripts/diagnostics/residual_similarity.py \\
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

EPS = 1e-6

# Pre-registered gate margins (docs/decisions.md 2026-06-12).
GATE_INTRA_MARGIN = 0.05   # median(intra) must sit this far below median(inter_task)
GATE_TASK_MARGIN = 0.02    # median(same_task) - median(inter_task) must be >= this


# --------------------------------------------------------------------------- #
# Residual construction
# --------------------------------------------------------------------------- #
def residual_raw(E: np.ndarray, variant: str) -> np.ndarray:
    """Un-normalised per-frame residual (T, D) for the requested anchor."""
    if variant == "raw":
        return E
    if variant == "A":
        return E - E[0:1]
    if variant == "B":
        return E - E.mean(axis=0, keepdims=True)
    raise ValueError(f"unknown variant {variant!r}")


def normalized_residuals(E: np.ndarray, variant: str):
    """(Rn, valid): per-frame residual directions, zeroed where norm < EPS."""
    R = residual_raw(E, variant)
    norms = np.linalg.norm(R, axis=1)
    valid = norms >= EPS
    Rn = np.zeros_like(R)
    Rn[valid] = R[valid] / norms[valid, None]
    return Rn, valid


def episode_mean_residual(E: np.ndarray, variant: str):
    """Net displacement normalize(mean_t (e_t - anchor)); None if degenerate."""
    m = residual_raw(E, variant).mean(axis=0)
    n = float(np.linalg.norm(m))
    if n < EPS:
        return None
    return m / n


# --------------------------------------------------------------------------- #
# Distributions (residual analogues of similarity_distributions.py)
# --------------------------------------------------------------------------- #
def intra_episode_residual_cosines(bundle: Bundle, variant: str,
                                   max_pairs_per_ep: int = 2000,
                                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Pooled within-episode pairwise cosines between residual directions."""
    rng = rng or np.random.default_rng(0)
    vals = []
    for ep in bundle.episode_indices:
        Rn, valid = normalized_residuals(bundle.frames(ep), variant)
        Rn = Rn[valid]                              # drop zero-norm residuals
        T = Rn.shape[0]
        if T < 2:
            continue
        S = Rn @ Rn.T
        iu = np.triu_indices(T, k=1)
        v = S[iu]
        if v.size > max_pairs_per_ep:
            v = rng.choice(v, size=max_pairs_per_ep, replace=False)
        vals.append(v)
    return np.concatenate(vals) if vals else np.array([])


def mean_pair_residual_cosines(bundle: Bundle, variant: str):
    """(same_task, diff_task, n_undef) cosines between episode-mean residuals."""
    vecs, tasks, n_undef = [], [], 0
    for ep in bundle.episode_indices:
        rm = episode_mean_residual(bundle.frames(ep), variant)
        if rm is None:
            n_undef += 1
            continue
        vecs.append(rm)
        tasks.append(bundle.ep_task[ep])
    if len(vecs) < 2:
        return np.array([]), np.array([]), n_undef
    M = np.stack(vecs)
    tasks = np.array(tasks)
    S = M @ M.T
    iu = np.triu_indices(len(vecs), k=1)
    same = tasks[iu[0]] == tasks[iu[1]]
    cos = S[iu]
    return cos[same], cos[~same], n_undef


def summary_row(variant: str, name: str, v: np.ndarray) -> dict:
    q = np.percentile(v, [5, 25, 50, 75, 95]) if v.size else [np.nan] * 5
    return {
        "variant": variant,
        "distribution": name,
        "n": int(v.size),
        "mean": float(np.mean(v)) if v.size else float("nan"),
        "std": float(np.std(v)) if v.size else float("nan"),
        "p05": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
        "p75": float(q[3]), "p95": float(q[4]),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    print(f"Loaded bundle: {len(b.episodes)} episodes, "
          f"{len(set(b.ep_task.values()))} tasks")

    rng = np.random.default_rng(0)
    variants = ["raw", "A", "B"]
    rows = []
    dists = {}  # (variant, kind) -> array, for the figure
    for var in variants:
        intra = intra_episode_residual_cosines(b, var, rng=rng)
        same, diff, n_undef = mean_pair_residual_cosines(b, var)
        dists[(var, "intra")] = intra
        dists[(var, "same")] = same
        dists[(var, "diff")] = diff
        rows.append(summary_row(var, "intra_episode_frame_pairs", intra))
        rows.append(summary_row(var, "inter_episode_same_task_means", same))
        rows.append(summary_row(var, "inter_task_means", diff))
        tag = f"  [{n_undef} episode-mean residuals undefined]" if n_undef else ""
        print(f"variant {var:>3}: intra n={intra.size}  same n={same.size}  "
              f"diff n={diff.size}{tag}")

    df = pd.DataFrame(rows)

    # ---- pre-registered gate (variant A) --------------------------------- #
    def med(a):
        return float(np.median(a)) if a.size else float("nan")

    m_intra = med(dists[("A", "intra")])
    m_same = med(dists[("A", "same")])
    m_inter = med(dists[("A", "diff")])
    cond_intra = m_intra <= m_inter - GATE_INTRA_MARGIN
    cond_task = (m_same - m_inter) >= GATE_TASK_MARGIN
    gate = "PASS" if (cond_intra and cond_task) else "FAIL"

    gate_lines = [
        "Pre-registered gate (variant A; docs/decisions.md 2026-06-12):",
        f"  median(intra_A)      = {m_intra:.3f}",
        f"  median(same_task_A)  = {m_same:.3f}",
        f"  median(inter_task_A) = {m_inter:.3f}",
        f"  cond1  intra <= inter_task - {GATE_INTRA_MARGIN}:  "
        f"{m_intra:.3f} <= {m_inter - GATE_INTRA_MARGIN:.3f}  -> {cond_intra}",
        f"  cond2  same - inter_task >= {GATE_TASK_MARGIN}:    "
        f"{m_same - m_inter:+.3f} >= {GATE_TASK_MARGIN}  -> {cond_task}",
        f"  VERDICT: {gate}",
    ]

    # ---- write tables ---------------------------------------------------- #
    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "tables" / "residual_similarity.csv", index=False)
    _write_md(df, out / "tables" / "residual_similarity.md", note=gate_lines)

    # ---- figure: one panel per variant ----------------------------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharex=True, sharey=True)
    bins = np.linspace(-0.5, 1.0, 90)
    titles = {"raw": "raw embeddings (§5.1)",
              "A": "residual A:  e_t - e_0",
              "B": "residual B:  e_t - mean(e)"}
    for ax, var in zip(axes, variants):
        for kind, label, color in [
            ("intra", "intra-episode (frame pairs)", "#1f77b4"),
            ("same", "inter-episode, same task (means)", "#2ca02c"),
            ("diff", "inter-task (means)", "#d62728"),
        ]:
            v = dists[(var, kind)]
            if v.size:
                ax.hist(v, bins=bins, density=True, alpha=0.55, label=label, color=color)
                ax.axvline(np.median(v), color=color, lw=1.5, ls="--")
        ax.set_title(titles[var], fontsize=10)
        ax.set_xlabel("cosine similarity")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Residual-space similarity distributions  "
                 f"(gate on variant A: {gate})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(out / "plots" / f"fig_residual_similarity.{ext}", dpi=150)
    plt.close(fig)

    print("\n" + df.to_string(index=False))
    print("\n" + "\n".join(gate_lines))
    print("\nwrote residual_similarity.{md,csv} and fig_residual_similarity.*")


def _write_md(df: pd.DataFrame, path: Path, note: list[str] | None = None) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    lines = [head, sep, *body]
    if note:
        lines += ["", "```", *note, "```"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
