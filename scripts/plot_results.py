"""
plot_results.py

Generate analysis figures from the experiment outputs.

Figures
-------
  fig1_accuracy_vs_cr.pdf   — Top-1 and Top-5 retrieval accuracy vs CR
  fig2_clipsim_vs_cr.pdf    — CLIP text-image similarity vs CR
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
from pathlib import Path
from typing import Optional

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

COLORS = {
    "uniform":       "#1f77b4",   # blue
    "random":        "#ff7f0e",   # orange
    "optical_flow":  "#2ca02c",   # green
    "attention_dino": "#9467bd",  # purple
}
LABELS = {
    "uniform":       "Uniform",
    "random":        "Random (mean ± std, 3 seeds)",
    "optical_flow":  "OpticalFlow (RAFT-Small)",
    "attention_dino": "Attention (DINOv2-S)",
}

K_SWEEP = [4, 8, 16, 32]


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
# Mock data (--mock flag, for local preview before real results exist)
# ---------------------------------------------------------------------------

def make_mock_eval() -> dict:
    rng = np.random.default_rng(0)
    results = {}

    # Uniform: monotonically improving accuracy with K
    for i, k in enumerate(K_SWEEP):
        base_cr = k / 150
        results[f"uniform_k{k}"] = {
            "top_1":    0.52 + i * 0.06 + rng.normal(0, 0.01),
            "top_5":    0.75 + i * 0.04 + rng.normal(0, 0.01),
            "clip_sim": 0.20 + i * 0.012 + rng.normal(0, 0.005),
            "mean_cr":  base_cr,
            "mean_n_kf": float(k),
        }

    # Random: slightly below uniform, with std
    for i, k in enumerate(K_SWEEP):
        base_cr = k / 150
        results[f"random_k{k}"] = {
            "top_1":     0.46 + i * 0.055 + rng.normal(0, 0.01),
            "top_1_std": 0.018 + rng.uniform(0, 0.008),
            "top_5":     0.70 + i * 0.038 + rng.normal(0, 0.01),
            "top_5_std": 0.015 + rng.uniform(0, 0.005),
            "clip_sim":  0.18 + i * 0.010 + rng.normal(0, 0.004),
            "clip_sim_std": 0.012,
            "mean_cr":   base_cr,
            "mean_n_kf": float(k),
        }

    # Optical flow: natural K around 12, good performance
    results["optical_flow"] = {
        "top_1":    0.74 + rng.normal(0, 0.01),
        "top_5":    0.91 + rng.normal(0, 0.01),
        "clip_sim": 0.246 + rng.normal(0, 0.005),
        "mean_cr":  12 / 150,
        "mean_n_kf": 12.3,
    }

    # Attention DINOv2: natural K around 14, best performance
    results["attention_dino"] = {
        "top_1":    0.79 + rng.normal(0, 0.01),
        "top_5":    0.93 + rng.normal(0, 0.01),
        "clip_sim": 0.261 + rng.normal(0, 0.005),
        "mean_cr":  14 / 150,
        "mean_n_kf": 14.1,
    }

    return {
        "config": {"K_sweep": K_SWEEP, "random_seeds": [42, 123, 456]},
        "results": results,
    }


def make_mock_perdemo() -> list[dict]:
    rng = np.random.default_rng(1)
    records = []
    T_mean = 150

    for task_id in range(10):
        for ep_idx in range(40):
            T = int(rng.integers(100, 220))
            split = "gallery" if ep_idx < 32 else "query"

            for k in K_SWEEP:
                for method in ("uniform", "random"):
                    records.append({
                        "extractor": f"{method}_k{k}_s42" if method == "random" else f"{method}_k{k}",
                        "task_id": task_id,
                        "episode_index": ep_idx,
                        "split": split,
                        "n_kf": k,
                        "cr":   k / T,
                        "T":    T,
                        "clip_sim": float(rng.normal(0.22, 0.04)),
                    })

            # Optical flow: variable K
            kf_of = int(np.clip(rng.normal(12, 4), 3, 35))
            records.append({
                "extractor": "optical_flow",
                "task_id": task_id, "episode_index": ep_idx, "split": split,
                "n_kf": kf_of, "cr": kf_of / T, "T": T,
                "clip_sim": float(rng.normal(0.245, 0.04)),
            })

            # Attention: variable K, slightly higher
            kf_att = int(np.clip(rng.normal(14, 5), 3, 40))
            records.append({
                "extractor": "attention_dino",
                "task_id": task_id, "episode_index": ep_idx, "split": split,
                "n_kf": kf_att, "cr": kf_att / T, "T": T,
                "clip_sim": float(rng.normal(0.260, 0.04)),
            })

    return records


def make_mock_consistency() -> dict:
    rng = np.random.default_rng(2)
    out = {}
    tasks = [f"task_{i}" for i in range(10)]
    ext_params = {
        "uniform_10":    (10.0, 0.0),
        "random_10":     (10.0, 0.5),
        "optical_flow":  (12.0, 4.0),
        "attention_dino": (14.0, 5.0),
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
# Figure 1: Retrieval accuracy vs compression ratio
# ---------------------------------------------------------------------------

def fig_accuracy_vs_cr(results: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle("Retrieval Accuracy vs Compression Ratio", fontsize=13, y=1.01)

    for ax, metric, ylabel in [
        (axes[0], "top_1", "Top-1 Accuracy"),
        (axes[1], "top_5", "Top-5 Accuracy"),
    ]:
        # Uniform — line
        crs  = [results[f"uniform_k{k}"]["mean_cr"] for k in K_SWEEP]
        vals = [results[f"uniform_k{k}"][metric]     for k in K_SWEEP]
        ax.plot(crs, vals, "o-", color=COLORS["uniform"],
                label=LABELS["uniform"], linewidth=2, markersize=6)
        for k, cr, v in zip(K_SWEEP, crs, vals):
            ax.annotate(f"K={k}", (cr, v), textcoords="offset points",
                        xytext=(4, 4), fontsize=8, color=COLORS["uniform"])

        # Random — line + shaded std band
        crs   = [results[f"random_k{k}"]["mean_cr"]       for k in K_SWEEP]
        vals  = [results[f"random_k{k}"][metric]           for k in K_SWEEP]
        stds  = [results[f"random_k{k}"].get(f"{metric}_std", 0.0) for k in K_SWEEP]
        ax.plot(crs, vals, "s--", color=COLORS["random"],
                label=LABELS["random"], linewidth=2, markersize=6)
        ax.fill_between(crs,
                         [v - s for v, s in zip(vals, stds)],
                         [v + s for v, s in zip(vals, stds)],
                         color=COLORS["random"], alpha=0.15)

        # OpticalFlow — single star marker
        of = results.get("optical_flow", {})
        if of:
            ax.plot(of["mean_cr"], of[metric], "*",
                    color=COLORS["optical_flow"], markersize=14,
                    label=LABELS["optical_flow"], zorder=5)

        # Attention — single star marker
        att = results.get("attention_dino", {})
        if att:
            ax.plot(att["mean_cr"], att[metric], "D",
                    color=COLORS["attention_dino"], markersize=9,
                    label=LABELS["attention_dino"], zorder=5)

        ax.set_xlabel("Compression Ratio (keyframes / T)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1.05)

    axes[0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_accuracy_vs_cr")


# ---------------------------------------------------------------------------
# Figure 2: CLIP similarity vs compression ratio
# ---------------------------------------------------------------------------

def fig_clipsim_vs_cr(results: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_title("CLIP Text-Image Similarity vs Compression Ratio", fontsize=12)

    crs  = [results[f"uniform_k{k}"]["mean_cr"]  for k in K_SWEEP]
    vals = [results[f"uniform_k{k}"]["clip_sim"] for k in K_SWEEP]
    ax.plot(crs, vals, "o-", color=COLORS["uniform"],
            label=LABELS["uniform"], linewidth=2, markersize=6)

    crs   = [results[f"random_k{k}"]["mean_cr"]         for k in K_SWEEP]
    vals  = [results[f"random_k{k}"]["clip_sim"]         for k in K_SWEEP]
    stds  = [results[f"random_k{k}"].get("clip_sim_std", 0.0) for k in K_SWEEP]
    ax.plot(crs, vals, "s--", color=COLORS["random"],
            label=LABELS["random"], linewidth=2, markersize=6)
    ax.fill_between(crs,
                     [v - s for v, s in zip(vals, stds)],
                     [v + s for v, s in zip(vals, stds)],
                     color=COLORS["random"], alpha=0.15)

    of = results.get("optical_flow", {})
    if of:
        ax.plot(of["mean_cr"], of["clip_sim"], "*",
                color=COLORS["optical_flow"], markersize=14,
                label=LABELS["optical_flow"], zorder=5)

    att = results.get("attention_dino", {})
    if att:
        ax.plot(att["mean_cr"], att["clip_sim"], "D",
                color=COLORS["attention_dino"], markersize=9,
                label=LABELS["attention_dino"], zorder=5)

    ax.set_xlabel("Compression Ratio (keyframes / T)")
    ax.set_ylabel("CLIP cosine similarity (image vs task text)")
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_clipsim_vs_cr")


# ---------------------------------------------------------------------------
# Figure 3: Per-demo keyframe count distribution
# ---------------------------------------------------------------------------

def fig_kf_distribution(records: list[dict], out_dir: Path) -> None:
    # Collect n_kf per extractor, grouped into display buckets
    buckets: dict[str, list[int]] = {
        "uniform_k4": [], "uniform_k8": [], "uniform_k16": [], "uniform_k32": [],
        "random_k8":  [],
        "optical_flow": [],
        "attention_dino": [],
    }
    for rec in records:
        ext = rec["extractor"]
        # Average random seeds into one bucket
        for k in K_SWEEP:
            if ext.startswith(f"random_k{k}_"):
                buckets.setdefault(f"random_k{k}", []).append(rec["n_kf"])
        if ext in buckets:
            buckets[ext].append(rec["n_kf"])

    # Only keep buckets with data
    buckets = {k: v for k, v in buckets.items() if v}

    labels_display = {
        "uniform_k4":    "Uniform K=4",
        "uniform_k8":    "Uniform K=8",
        "uniform_k16":   "Uniform K=16",
        "uniform_k32":   "Uniform K=32",
        "random_k4":     "Random K=4",
        "random_k8":     "Random K=8",
        "random_k16":    "Random K=16",
        "random_k32":    "Random K=32",
        "optical_flow":  "OpticalFlow",
        "attention_dino": "Attention",
    }
    order = [k for k in labels_display if k in buckets]
    data  = [buckets[k] for k in order]
    names = [labels_display[k] for k in order]

    colors_per_box = []
    for k in order:
        if k.startswith("uniform"):     colors_per_box.append(COLORS["uniform"])
        elif k.startswith("random"):    colors_per_box.append(COLORS["random"])
        elif k == "optical_flow":       colors_per_box.append(COLORS["optical_flow"])
        else:                           colors_per_box.append(COLORS["attention_dino"])

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.set_title("Keyframe Count Distribution Across Episodes", fontsize=12)

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors_per_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ("whiskers", "caps", "fliers"):
        for item, color in zip(bp[element], np.repeat(colors_per_box, 2)):
            item.set_color(color)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Number of keyframes selected (n_kf)")

    legend_patches = [mpatches.Patch(color=COLORS[k], label=LABELS[k], alpha=0.7)
                      for k in ("uniform", "random", "optical_flow", "attention_dino")]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, out_dir, "fig3_kf_distribution")


# ---------------------------------------------------------------------------
# Figure 4: Keyframe-count CV across tasks
# ---------------------------------------------------------------------------

def fig_consistency(consistency: dict, out_dir: Path) -> None:
    # Collect cv_kf per extractor across all tasks
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
        if "attention" in name or "dino" in name: return COLORS["attention_dino"]
        return "#888888"

    bar_colors = [_color(e) for e in ext_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("Keyframe-Count Consistency Across Tasks\n"
                 "(CV = std / mean; lower = more consistent)", fontsize=11)
    xs = range(len(ext_names))
    bars = ax.bar(xs, means, yerr=stds, color=bar_colors, alpha=0.75,
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

    order = (
        [f"uniform_k{k}" for k in K_SWEEP]
        + [f"random_k{k}"  for k in K_SWEEP]
        + ["optical_flow", "attention_dino"]
    )
    for label in order:
        r = results.get(label)
        if r is None:
            continue
        print(
            f"{label:<24} {r['top_1']:>6.3f} {r.get('top_1_std', 0):>5.3f} "
            f"{r['top_5']:>6.3f} {r.get('top_5_std', 0):>5.3f} "
            f"{r['clip_sim']:>9.4f} {r['mean_cr']:>8.4f} {r['mean_n_kf']:>8.1f}"
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
        perdemo     = load_perdemo(args.perdemo)
        consistency = (load_consistency(args.consistency)
                       if Path(args.consistency).exists() else {})

    results = eval_data["results"]

    print_summary(results)

    print(f"\nGenerating figures → {out_dir}/")
    fig_accuracy_vs_cr(results, out_dir)
    fig_clipsim_vs_cr(results, out_dir)
    fig_kf_distribution(perdemo, out_dir)
    if consistency:
        fig_consistency(consistency, out_dir)
    else:
        print("  (no consistency file — skipping fig4)")

    print("\nDone.")


if __name__ == "__main__":
    main()
