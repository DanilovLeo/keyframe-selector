"""
plot_results.py

Generate analysis figures from the experiment outputs.

Schema
------
Since the rank-to-K change, EVERY method in eval_retrieval.json["results"] is
keyed per-K: ``<method>_k<K>`` — e.g. uniform_k4, random_k8, optical_flow_k16,
attention_k32, frame_diff_k4. The aggregated random entry ``random_k<K>`` holds
the mean ± std across seeds; the per-seed entries ``random_k<K>_s<seed>`` are
also present but are skipped here (their trailing ``_s<seed>`` excludes them
from the curve parser). Method + K are parsed generically (split on the
trailing ``_k<int>``) so adding/removing methods or K values never re-breaks
this script.

Figures
-------
  fig1_accuracy_vs_cr.pdf   — Top-1 and Top-5 retrieval accuracy vs CR
                              (one curve per method, points ordered by K)
  fig2_clipsim_vs_cr.pdf    — CLIP text-image similarity vs CR (per-method)
  fig3_kf_distribution.pdf  — Per-demo keyframe count box plots
  fig4_consistency.pdf      — Keyframe-count CV across tasks per extractor

Usage
-----
    # With real results:
    python scripts/plot_results.py \
        --eval    results/eval_retrieval.json \
        --perdemo results/eval_per_demo.jsonl \
        --consistency results/consistency_check_bridge.json \
        --out_dir results/figures

    # Preview figure layout locally (no data needed):
    python scripts/plot_results.py --mock --out_dir results/figures
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Visual style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})

# Keyed by parsed METHOD base name (the part before the trailing _k<int>).
COLORS = {
    "uniform":      "#1f77b4",   # blue
    "random":       "#ff7f0e",   # orange
    "optical_flow": "#2ca02c",   # green
    "attention":    "#9467bd",   # purple
    "frame_diff":   "#d62728",   # red
}
LABELS = {
    "uniform":      "Uniform",
    "random":       "Random (mean ± std, 3 seeds)",
    "optical_flow": "OpticalFlow (RAFT-Small)",
    "attention":    "Attention (DINOv2-S)",
    "frame_diff":   "FrameDiff (pixel MAD)",
}
MARKERS = {
    "uniform":      "o",
    "random":       "s",
    "optical_flow": "^",
    "attention":    "D",
    "frame_diff":   "v",
}
# Display/plot order; any unknown method is appended after these, alphabetically.
METHOD_ORDER = ["uniform", "random", "optical_flow", "attention", "frame_diff"]
_FALLBACK_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Result keys that denote an "all frames" ceiling (drawn as a horizontal line
# if present). The real eval output does not emit one today; this is purely
# conditional so a future ceiling entry renders without code changes.
_CEILING_KEYS = ("all_frames", "all", "full", "ceiling", "upper_bound")

K_SWEEP = [4, 8, 16, 32]   # used only by the --mock generators


# ---------------------------------------------------------------------------
# Generic <method>_k<K> parsing
# ---------------------------------------------------------------------------

# Anchored at end: 'random_k4_s42' (trailing _s42) deliberately does NOT match,
# so per-seed entries are skipped in favour of the aggregated 'random_k4'.
_KEY_RE = re.compile(r"^(?P<method>.+)_k(?P<k>\d+)$")


def parse_method_k(key: str):
    """Parse 'optical_flow_k16' -> ('optical_flow', 16). None if it doesn't match."""
    m = _KEY_RE.match(key)
    if m is None:
        return None
    return m.group("method"), int(m.group("k"))


def group_curves(results: dict) -> dict:
    """Group result entries into per-method curves.

    Returns {method: [(K, entry), ...]} with each list sorted by K ascending.
    Per-seed random entries and any non-<method>_k<int> keys (e.g. a ceiling)
    are ignored here.
    """
    curves: dict[str, list] = {}
    for key, entry in results.items():
        parsed = parse_method_k(key)
        if parsed is None:
            continue
        method, k = parsed
        curves.setdefault(method, []).append((k, entry))
    for method in curves:
        curves[method].sort(key=lambda kv: kv[0])
    return curves


def _ordered_methods(curves: dict) -> list:
    known = [m for m in METHOD_ORDER if m in curves]
    extra = sorted(m for m in curves if m not in METHOD_ORDER)
    return known + extra


def _style(method: str, idx: int):
    color = COLORS.get(method, _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)])
    marker = MARKERS.get(method, "o")
    label = LABELS.get(method, method)
    return color, marker, label


def find_ceiling(results: dict):
    """Return the all-frames ceiling entry if one is present, else None."""
    for key in _CEILING_KEYS:
        if key in results:
            return results[key]
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_perdemo(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_consistency(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Mock data (--mock flag) — emitted in the SAME schema as the real output
# ---------------------------------------------------------------------------

def make_mock_eval() -> dict:
    """Synthetic eval_retrieval.json: per-K curves for every method (real schema)."""
    rng = np.random.default_rng(0)
    results = {}

    # Matched-budget methods (uniform + the three heuristics): per-K entries.
    # (base_top1, slope_top1, base_top5, slope_top5, base_clip, slope_clip)
    profiles = {
        "uniform":      (0.52, 0.060, 0.75, 0.040, 0.200, 0.012),
        "optical_flow": (0.56, 0.064, 0.78, 0.040, 0.215, 0.013),
        "attention":    (0.60, 0.066, 0.80, 0.038, 0.225, 0.013),
        "frame_diff":   (0.50, 0.060, 0.74, 0.040, 0.205, 0.012),
    }
    for method, (b1, s1, b5, s5, bc, sc) in profiles.items():
        for i, k in enumerate(K_SWEEP):
            results[f"{method}_k{k}"] = {
                "top_1":     float(min(0.99, b1 + i * s1 + rng.normal(0, 0.01))),
                "top_5":     float(min(0.99, b5 + i * s5 + rng.normal(0, 0.01))),
                "clip_sim":  float(bc + i * sc + rng.normal(0, 0.004)),
                "mean_cr":   k / 150,
                "mean_n_kf": float(k),
            }

    # Random: aggregated per-K entry with mean ± std across seeds.
    for i, k in enumerate(K_SWEEP):
        results[f"random_k{k}"] = {
            "top_1":        float(0.46 + i * 0.055 + rng.normal(0, 0.01)),
            "top_1_std":    float(0.018 + rng.uniform(0, 0.008)),
            "top_5":        float(0.70 + i * 0.038 + rng.normal(0, 0.01)),
            "top_5_std":    float(0.015 + rng.uniform(0, 0.005)),
            "clip_sim":     float(0.18 + i * 0.010 + rng.normal(0, 0.004)),
            "clip_sim_std": 0.012,
            "mean_cr":      k / 150,
            "mean_n_kf":    float(k),
        }

    return {
        "config": {"K_sweep": K_SWEEP, "random_seeds": [42, 123, 456]},
        "results": results,
    }


def make_mock_perdemo() -> list[dict]:
    """Synthetic per-demo records using the real per-K extractor labels.

    All extractors are matched-budget now, so n_kf == K for every record.
    """
    rng = np.random.default_rng(1)
    records = []
    matched_methods = ("uniform", "optical_flow", "attention", "frame_diff")

    for task_id in range(10):
        for ep_idx in range(40):
            T = int(rng.integers(100, 220))
            split = "gallery" if ep_idx < 32 else "query"
            for k in K_SWEEP:
                for method in matched_methods:
                    records.append({
                        "extractor": f"{method}_k{k}",
                        "task_id": task_id, "episode_index": ep_idx, "split": split,
                        "n_kf": k, "cr": k / T, "T": T,
                        "clip_sim": float(rng.normal(0.23, 0.04)),
                    })
                for seed in (42, 123, 456):
                    records.append({
                        "extractor": f"random_k{k}_s{seed}",
                        "task_id": task_id, "episode_index": ep_idx, "split": split,
                        "n_kf": k, "cr": k / T, "T": T,
                        "clip_sim": float(rng.normal(0.21, 0.04)),
                    })
    return records


def make_mock_consistency() -> dict:
    rng = np.random.default_rng(2)
    out = {}
    tasks = [f"task_{i}" for i in range(10)]
    # Variable-N natural counts (run_consistency.py uses the n=None extractors).
    ext_params = {
        "uniform_10":    (10.0, 0.0),
        "random_10":     (10.0, 0.5),
        "optical_flow":  (12.0, 4.0),
        "attention_dino": (14.0, 5.0),
        "frame_diff":    (13.0, 4.5),
    }
    for task in tasks:
        out[task] = {"task": task, "n_episodes": 40, "extractors": {}}
        for ext, (mean, std) in ext_params.items():
            counts = rng.normal(mean, std, 40).clip(2)
            m, s = counts.mean(), counts.std()
            out[task]["extractors"][ext] = {
                "mean_kf": float(m),
                "std_kf":  float(s),
                "cv_kf":   float(s / m),
                "mean_cr": float(m / 150),
            }
    return out


# ---------------------------------------------------------------------------
# Figure 1: Retrieval accuracy vs compression ratio (per-method curves)
# ---------------------------------------------------------------------------

def _plot_metric_curves(ax, curves: dict, metric: str, ceiling=None) -> None:
    """Plot one curve per method: *metric* vs mean_cr, points ordered by K."""
    for idx, method in enumerate(_ordered_methods(curves)):
        pts = curves[method]
        vals = [e.get(metric) for _, e in pts]
        if any(v is None for v in vals):
            continue  # method didn't record this metric
        ks  = [k for k, _ in pts]
        crs = [e["mean_cr"] for _, e in pts]
        color, marker, label = _style(method, idx)
        linestyle = "--" if method == "random" else "-"
        ax.plot(crs, vals, marker=marker, linestyle=linestyle, color=color,
                label=label, linewidth=2, markersize=6)

        # Shaded ± std band where the method records it (e.g. random).
        stds = [e.get(f"{metric}_std") for _, e in pts]
        if all(s is not None for s in stds):
            ax.fill_between(crs,
                            [v - s for v, s in zip(vals, stds)],
                            [v + s for v, s in zip(vals, stds)],
                            color=color, alpha=0.15)

        # Annotate each point with its K.
        for k, cr, v in zip(ks, crs, vals):
            ax.annotate(f"K={k}", (cr, v), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=color)

    if ceiling is not None and ceiling.get(metric) is not None:
        ax.axhline(ceiling[metric], color="#444444", linestyle=":", linewidth=1.5,
                   label="All frames (ceiling)")


def fig_accuracy_vs_cr(results: dict, out_dir: Path) -> None:
    curves = group_curves(results)
    ceiling = find_ceiling(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle("Retrieval Accuracy vs Compression Ratio", fontsize=13, y=1.01)

    for ax, metric, ylabel in [
        (axes[0], "top_1", "Top-1 Accuracy"),
        (axes[1], "top_5", "Top-5 Accuracy"),
    ]:
        _plot_metric_curves(ax, curves, metric, ceiling)
        ax.set_xlabel("Compression Ratio (keyframes / T)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

    axes[0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_accuracy_vs_cr")


# ---------------------------------------------------------------------------
# Figure 2: CLIP similarity vs compression ratio (per-method curves)
# ---------------------------------------------------------------------------

def fig_clipsim_vs_cr(results: dict, out_dir: Path) -> None:
    curves = group_curves(results)
    ceiling = find_ceiling(results)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("CLIP Text-Image Similarity vs Compression Ratio", fontsize=12)

    _plot_metric_curves(ax, curves, "clip_sim", ceiling)

    ax.set_xlabel("Compression Ratio (keyframes / T)")
    ax.set_ylabel("CLIP cosine similarity (image vs task text)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_clipsim_vs_cr")


# ---------------------------------------------------------------------------
# Figure 3: Per-demo keyframe count distribution
# ---------------------------------------------------------------------------

def fig_kf_distribution(records: list[dict], out_dir: Path) -> None:
    # Bucket n_kf by <method>_k<K>; collapse random seeds into one bucket per K.
    seed_re = re.compile(r"^(?P<base>random_k\d+)_s\d+$")
    buckets: dict[str, list[int]] = {}
    for rec in records:
        ext = rec["extractor"]
        m = seed_re.match(ext)
        key = m.group("base") if m else ext
        if parse_method_k(key) is None:
            continue
        buckets.setdefault(key, []).append(rec["n_kf"])

    buckets = {k: v for k, v in buckets.items() if v}
    if not buckets:
        print("  (no per-demo data — skipping fig3)")
        return

    def _sort_key(key: str):
        method, k = parse_method_k(key)
        mi = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
        return (mi, k)

    order = sorted(buckets, key=_sort_key)
    data  = [buckets[key] for key in order]

    names, box_colors, methods_present = [], [], []
    for idx, key in enumerate(order):
        method, k = parse_method_k(key)
        short = LABELS.get(method, method).split(" ")[0]
        names.append(f"{short} K={k}")
        box_colors.append(_style(method, idx)[0])
        methods_present.append(method)

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 4.5))
    ax.set_title("Keyframe Count Distribution Across Episodes\n"
                 "(matched-budget extractors select exactly K)", fontsize=11)

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ("whiskers", "caps", "fliers"):
        for item, color in zip(bp[element], np.repeat(box_colors, 2)):
            item.set_color(color)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of keyframes selected (n_kf)")

    seen = []
    for method in _ordered_methods({m: None for m in methods_present}):
        if method not in seen:
            seen.append(method)
    legend_patches = [
        mpatches.Patch(color=COLORS.get(m, "#888888"), label=LABELS.get(m, m), alpha=0.7)
        for m in seen
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_kf_distribution")


# ---------------------------------------------------------------------------
# Figure 4: Keyframe-count CV across tasks
# ---------------------------------------------------------------------------

def fig_consistency(consistency: dict, out_dir: Path) -> None:
    ext_cv: dict[str, list[float]] = {}
    for task_data in consistency.values():
        for ext, stats in task_data.get("extractors", {}).items():
            ext_cv.setdefault(ext, []).append(stats["cv_kf"])

    if not ext_cv:
        print("  (consistency data empty — skipping fig4)")
        return

    ext_names = sorted(ext_cv.keys())
    means = [np.mean(ext_cv[e]) for e in ext_names]
    stds  = [np.std(ext_cv[e])  for e in ext_names]

    def _color(name: str) -> str:
        if "uniform" in name: return COLORS["uniform"]
        if "random"  in name: return COLORS["random"]
        if "optical" in name: return COLORS["optical_flow"]
        if "attention" in name or "dino" in name: return COLORS["attention"]
        if "frame" in name or "diff" in name: return COLORS["frame_diff"]
        return "#888888"

    bar_colors = [_color(e) for e in ext_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Keyframe-Count Consistency Across Tasks\n"
                 "(CV = std / mean; lower = more consistent)", fontsize=11)
    xs = range(len(ext_names))
    ax.bar(xs, means, yerr=stds, color=bar_colors, alpha=0.75,
           capsize=4, error_kw=dict(elinewidth=1.2))
    ax.set_xticks(xs)
    ax.set_xticklabels(ext_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean CV (std / mean of keyframe counts)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_consistency")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results: dict) -> None:
    print("\n" + "=" * 82)
    print(f"{'Extractor':<24} {'Top-1':>6} {'±':>5} {'Top-5':>6} {'±':>5} "
          f"{'CLIP-sim':>9} {'mean_CR':>8} {'mean_KF':>8}")
    print("-" * 82)

    curves = group_curves(results)
    for method in _ordered_methods(curves):
        for k, r in curves[method]:
            label = f"{method}_k{k}"
            print(
                f"{label:<24} {r.get('top_1', 0):>6.3f} {r.get('top_1_std', 0):>5.3f} "
                f"{r.get('top_5', 0):>6.3f} {r.get('top_5_std', 0):>5.3f} "
                f"{r.get('clip_sim', 0):>9.4f} {r.get('mean_cr', 0):>8.4f} "
                f"{r.get('mean_n_kf', 0):>8.1f}"
            )
    print("=" * 82)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  saved {stem}.pdf / .png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",        default="results/eval_retrieval.json")
    parser.add_argument("--perdemo",     default="results/eval_per_demo.jsonl")
    parser.add_argument("--consistency", default="results/consistency_check_bridge.json")
    parser.add_argument("--out_dir",     default="results/figures")
    parser.add_argument("--mock",        action="store_true",
                        help="Use synthetic data to preview figure layout")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    if args.mock:
        print("Using MOCK data — figures show layout only, not real results.")
        eval_data    = make_mock_eval()
        perdemo      = make_mock_perdemo()
        consistency  = make_mock_consistency()
    else:
        eval_data   = load_eval(args.eval)
        perdemo     = (load_perdemo(args.perdemo)
                       if Path(args.perdemo).exists() else [])
        consistency = (load_consistency(args.consistency)
                       if Path(args.consistency).exists() else {})

    results = eval_data["results"]

    print_summary(results)

    print(f"\nGenerating figures → {out_dir}/")
    fig_accuracy_vs_cr(results, out_dir)
    fig_clipsim_vs_cr(results, out_dir)
    if perdemo:
        fig_kf_distribution(perdemo, out_dir)
    else:
        print("  (no per-demo file — skipping fig3)")
    if consistency:
        fig_consistency(consistency, out_dir)
    else:
        print("  (no consistency file — skipping fig4)")

    print("\nDone.")


if __name__ == "__main__":
    main()
