"""
crossover_analysis.py  (Task 3 — explains the §5.6 K=32 coverage crossover)

§5.6 reported a crossover in embedding-space coverage error: even spacing
(`uniform`) is best at tight budgets, but content-adaptive anchors
(`attention`, `frame_diff`) overtake it at K=32.  This script tests the
candidate mechanism, pre-registered in docs/decisions.md (2026-06-12): adaptive
methods place their interior anchors in high embedding-**velocity** regions
(v_t = ||e_t - e_{t-1}||), so once the budget is generous they cover the
fast-changing parts of the trajectory — where nearest-anchor distance
concentrates — better than velocity-agnostic even spacing; at tight budgets that
is outweighed by the larger temporal holes their clustering leaves.

Per method x K (episode means over the 863 episodes):
  mean_cov          §5.6 coverage error (recomputed via episode_coverage; xcheck)
  mean_cov_highvel  coverage over non-selected frames with >= episode-median vel
  mean_cov_lowvel   coverage over non-selected frames with <  episode-median vel
  velocity_ratio    mean vel at interior anchors / mean vel over interior frames
  max_gap_ratio     largest inter-anchor temporal gap / mean gap

Pre-registered rule (decisions.md 2026-06-12), evaluated at K=32:
  SUPPORTS  -> add methods.md §5.6.1   if velocity_ratio(attention) and
               velocity_ratio(frame_diff) both exceed uniform's by >= 0.10 and
               are > 1.0, AND (uniform - method) coverage advantage is larger in
               the high-velocity region than the low-velocity region for both.
  INCONCLUSIVE -> record null in decisions.md, leave methods.md unchanged.

Outputs:
  results/tables/crossover_analysis.{md,csv}
  results/plots/fig_crossover_velocity.{pdf,png}

Pure numpy + matplotlib; reads only the bundle (needs exported keyframe indices).
No GPU.

Usage (from keyframe-selector/):
    python scripts/diagnostics/crossover_analysis.py \\
        --bundle results/bundle --out_dir results
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `bundle` import

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from bundle import Bundle              # noqa: E402
from coverage_error import episode_coverage  # noqa: E402  (exact §5.6 definition)


# --------------------------------------------------------------------------- #
# Per-episode measurements
# --------------------------------------------------------------------------- #
def frame_velocity(E: np.ndarray) -> np.ndarray:
    """Per-frame embedding speed ||e_t - e_{t-1}||; frame 0 mirrors edge 0."""
    T = E.shape[0]
    if T < 2:
        return np.zeros(T)
    g = np.linalg.norm(np.diff(E, axis=0), axis=1)   # (T-1,)
    gf = np.empty(T)
    gf[0] = g[0]
    gf[1:] = g
    return gf


def velocity_ratio(E: np.ndarray, sel) -> float:
    """mean vel at interior anchors / mean vel over interior frames; nan if undefined."""
    T = E.shape[0]
    if T < 3:
        return np.nan
    gf = frame_velocity(E)
    interior = np.arange(1, T - 1)
    base = float(gf[interior].mean())
    if base <= 1e-12:
        return np.nan
    sel = np.unique(np.asarray(sel, dtype=int))
    sel_int = sel[(sel > 0) & (sel < T - 1)]
    if sel_int.size == 0:
        return np.nan
    return float(gf[sel_int].mean() / base)


def max_gap_ratio(E: np.ndarray, sel) -> float:
    """Largest inter-anchor temporal gap / mean gap; nan if < 3 anchors."""
    sel = np.unique(np.asarray(sel, dtype=int))
    if sel.size < 3:
        return np.nan
    gaps = np.diff(sel)
    m = float(gaps.mean())
    if m <= 0:
        return np.nan
    return float(gaps.max() / m)


def coverage_by_velocity(E: np.ndarray, sel) -> tuple[float, float]:
    """(high_vel_cov, low_vel_cov): mean nearest-anchor distance split at median vel."""
    T = E.shape[0]
    sel = np.unique(np.asarray(sel, dtype=int))
    mask = np.ones(T, dtype=bool)
    mask[sel] = False
    if not mask.any():
        return np.nan, np.nan
    gf = frame_velocity(E)
    sims = E[mask] @ E[sel].T
    dmin = 1.0 - sims.max(axis=1)
    vel_nonsel = gf[mask]
    thr = float(np.median(gf))
    hi = vel_nonsel >= thr
    lo = ~hi
    hi_cov = float(dmin[hi].mean()) if hi.any() else np.nan
    lo_cov = float(dmin[lo].mean()) if lo.any() else np.nan
    return hi_cov, lo_cov


def per_episode_metrics(b: Bundle, idx_fn) -> dict:
    """Episode-mean of each metric for one index rule (nan-aware)."""
    cov, covhi, covlo, vr, gr = [], [], [], [], []
    for ep in b.episode_indices:
        E = b.frames(ep)
        sel = idx_fn(ep, E.shape[0])
        cov.append(episode_coverage(E, sel)[0])
        hi, lo = coverage_by_velocity(E, sel)
        covhi.append(hi); covlo.append(lo)
        vr.append(velocity_ratio(E, sel))
        gr.append(max_gap_ratio(E, sel))
    return {
        "mean_cov": np.array(cov, float),
        "mean_cov_highvel": np.array(covhi, float),
        "mean_cov_lowvel": np.array(covlo, float),
        "velocity_ratio": np.array(vr, float),
        "max_gap_ratio": np.array(gr, float),
    }


def _label_indices(b: Bundle, label: str):
    return lambda ep, T: b.indices(ep, label)


def method_metrics(b: Bundle, method: str, k: int) -> dict:
    """Per-episode metric arrays for a method at K (random averaged over seeds)."""
    if method == "random":
        seeds = [per_episode_metrics(b, _label_indices(b, f"random_k{k}_s{s}"))
                 for s in b.random_seeds]
        keys = seeds[0].keys()
        with warnings.catch_warnings():  # all-nan episodes -> nan (benign)
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {key: np.nanmean(np.stack([s[key] for s in seeds]), axis=0)
                    for key in keys}
    return per_episode_metrics(b, _label_indices(b, f"{method}_k{k}"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="results/bundle")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    b = Bundle(args.bundle)
    if not b.has_indices():
        sys.exit("Bundle has no keyframes.jsonl — re-export without --no_indices.")
    print(f"Loaded bundle: {len(b.episode_indices)} episodes")

    cols = ["mean_cov", "mean_cov_highvel", "mean_cov_lowvel",
            "velocity_ratio", "max_gap_ratio"]
    rows = []
    agg = {}  # (method, k) -> {col: scalar mean}
    for k in b.k_sweep:
        for m in b.methods:
            arr = method_metrics(b, m, k)
            means = {c: float(np.nanmean(arr[c])) for c in cols}
            agg[(m, k)] = means
            rows.append({"method": m, "K": k, **means})

    df = pd.DataFrame(rows)

    out = Path(args.out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tables" / "crossover_analysis.csv", index=False)

    # ---- pre-registered verdict (K=32) ----------------------------------- #
    kmax = max(b.k_sweep)
    uni = agg[("uniform", kmax)]
    verdict_lines = [
        f"Pre-registered mechanism test at K={kmax} "
        "(docs/decisions.md 2026-06-12):",
        f"  uniform  velocity_ratio = {uni['velocity_ratio']:.3f}",
    ]
    support_flags = []
    for m in ("attention", "frame_diff"):
        if (m, kmax) not in agg:
            continue
        a = agg[(m, kmax)]
        vr_margin = a["velocity_ratio"] - uni["velocity_ratio"]
        cond_vr = (a["velocity_ratio"] > 1.0) and (vr_margin >= 0.10)
        adv_hi = uni["mean_cov_highvel"] - a["mean_cov_highvel"]
        adv_lo = uni["mean_cov_lowvel"] - a["mean_cov_lowvel"]
        cond_loc = adv_hi > adv_lo
        support_flags.append(cond_vr and cond_loc)
        verdict_lines += [
            f"  {m:<11} velocity_ratio = {a['velocity_ratio']:.3f} "
            f"(margin {vr_margin:+.3f}; >1 & >=0.10 -> {cond_vr})",
            f"  {m:<11} hi-vel adv = {adv_hi:+.4f}  lo-vel adv = {adv_lo:+.4f} "
            f"(hi>lo -> {cond_loc})",
        ]
    verdict = "SUPPORTS" if (support_flags and all(support_flags)) else "INCONCLUSIVE"
    verdict_lines.append(f"  VERDICT: {verdict}")

    _write_md(df, out / "tables" / "crossover_analysis.md", note=verdict_lines)

    # ---- figure: crossover + velocity placement -------------------------- #
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.4))
    colors = {"uniform": "#1f77b4", "random": "#7f7f7f",
              "optical_flow": "#ff7f0e", "attention": "#2ca02c",
              "frame_diff": "#d62728"}
    ks = list(b.k_sweep)
    for m in b.methods:
        c = colors.get(m, None)
        ax0.plot(ks, [agg[(m, k)]["mean_cov"] for k in ks],
                 marker="o", label=m, color=c)
        ax1.plot(ks, [agg[(m, k)]["velocity_ratio"] for k in ks],
                 marker="o", label=m, color=c)
    ax0.set_xlabel("K"); ax0.set_ylabel("coverage error (lower = better)")
    ax0.set_title("Coverage error vs K (the §5.6 crossover)")
    ax0.set_xscale("log", base=2); ax0.set_xticks(ks); ax0.set_xticklabels(ks)
    ax1.axhline(1.0, color="k", lw=0.8, ls=":")
    ax1.set_xlabel("K"); ax1.set_ylabel("anchor velocity ratio (interior)")
    ax1.set_title("Do anchors land in fast regions?  (>1 = yes)")
    ax1.set_xscale("log", base=2); ax1.set_xticks(ks); ax1.set_xticklabels(ks)
    ax1.legend(fontsize=8, loc="best")
    fig.suptitle(f"K=32 coverage crossover mechanism — verdict: {verdict}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(out / "plots" / f"fig_crossover_velocity.{ext}", dpi=150)
    plt.close(fig)

    print("\n" + df.to_string(index=False))
    print("\n" + "\n".join(verdict_lines))
    print("\nwrote crossover_analysis.{md,csv} and fig_crossover_velocity.*")


def _write_md(df: pd.DataFrame, path: Path, note: list[str] | None = None) -> None:
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4f}"
        return str(v)

    body = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False)]
    banner = ("> Mechanism analysis for the §5.6 K=32 coverage crossover "
              "(velocity placement of anchors). See docs/decisions.md "
              "(2026-06-12).\n\n")
    lines = [banner + head, sep, *body]
    if note:
        lines += ["", "```", *note, "```"]
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
