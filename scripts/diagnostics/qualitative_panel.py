#!/usr/bin/env python3
"""Qualitative keyframe panel — one episode, the frames each method selects.

Renders, for a single episode, the frames every extractor picks at a fixed budget K
as a method x slot image grid, plus a rug-plot timeline of the selected indices over
0..T. This is a qualitative illustration of the coverage story (methods.md S5.6):
uniform spreads anchors evenly, while optical_flow / attention concentrate them on
low-motion "settled" frames and leave temporal gaps.

Scope: runs the existing extractors on already-decodable pixels and draws the indices
they return -- no new metric, no new data, no model fit, no robot-state signal
(decisions.md, 2026-06-14). Pure illustration.

Two modes:
  --candidates        list mid-length episodes (metadata only, no decode) to choose from
  --episode IDX       render the panel -> results/plots/fig_qualitative_panel.{pdf,png}

Needs the full environment (torch / torchvision / timm / av); CPU is enough for one
episode. optical_flow pulls RAFT-Small weights, attention pulls DINOv2 ViT-S/14.

Example:
  python scripts/diagnostics/qualitative_panel.py --candidates
  python scripts/diagnostics/qualitative_panel.py --episode 12345 --k 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))            # repo root, for `src.` imports
MODELS_CFG = ROOT / "configs" / "models.yaml"


def _timm_model() -> str:
    """DINOv2 backbone the attention extractor uses (pinned in configs/models.yaml)."""
    try:
        import yaml
        return yaml.safe_load(MODELS_CFG.read_text())["dinov2"]["timm_model"]
    except Exception:
        return "vit_small_patch14_dinov2"


def build_methods(k: int, seed: int):
    """The five selection strategies, instantiated exactly as the eval grid does."""
    from src.extractors import (UniformExtractor, RandomExtractor,
                                OpticalFlowExtractor, AttentionSaliencyExtractor,
                                FrameDiffExtractor)
    return [
        ("uniform",      UniformExtractor(n_keyframes=k)),
        ("random",       RandomExtractor(n_keyframes=k, seed=seed)),
        ("optical_flow", OpticalFlowExtractor(n_keyframes=k)),
        ("attention",    AttentionSaliencyExtractor(n_keyframes=k, timm_model=_timm_model())),
        ("frame_diff",   FrameDiffExtractor(n_keyframes=k)),
    ]


def list_candidates(loader, min_len: int, max_len: int, min_demos: int):
    """Mid-length episodes (cached metadata, no decode) for picking a clear clip."""
    rows = []
    for task in loader.list_tasks(min_demos=min_demos):
        for ep in loader.list_episodes(task):
            T = loader._episode_meta[ep]["length"]   # metadata only, no pixels
            if min_len <= T <= max_len:
                rows.append((ep, T, task))
    rows.sort(key=lambda r: (r[2], r[1]))
    return rows


def render(frames: np.ndarray, picks: dict, task: str, k: int, out_dir: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(frames)
    methods = list(picks)
    M = len(methods)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))[:M]

    fig = plt.figure(figsize=(2 * k, 2 * M + 1.8))
    gs = fig.add_gridspec(M + 1, k, height_ratios=[*([3] * M), 1.4],
                          hspace=0.4, wspace=0.05)

    # image grid: one row per method, one column per keyframe slot
    for r, m in enumerate(methods):
        idx = picks[m]
        for c in range(k):
            ax = fig.add_subplot(gs[r, c])
            if c < len(idx):
                ax.imshow(frames[idx[c]])
                ax.set_title(f"t={int(idx[c])}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.text(-0.2, 0.5, m, transform=ax.transAxes, ha="right", va="center",
                        fontsize=11, fontweight="bold", color=colors[r])

    # rug-plot timeline: where on 0..T each method placed its anchors
    axt = fig.add_subplot(gs[M, :])
    for r, m in enumerate(methods):
        axt.eventplot(np.asarray(picks[m], dtype=int), lineoffsets=M - 1 - r,
                      linelengths=0.8, colors=[colors[r]])
    axt.set_xlim(-0.5, T - 0.5)
    axt.set_ylim(-0.5, M - 0.5)
    axt.set_yticks(range(M))
    axt.set_yticklabels(methods[::-1], fontsize=9)
    axt.set_xlabel(f"frame index t   (T = {T})", fontsize=10)
    axt.set_title("temporal distribution of selected keyframes", fontsize=10)
    for sp in ("top", "right", "left"):
        axt.spines[sp].set_visible(False)

    fig.suptitle(f'Selected keyframes per method  —  K = {k},  task: "{task}"',
                 fontsize=12, y=1.0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_qualitative_panel.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_qualitative_panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out / "fig_qualitative_panel.png"


def main() -> None:
    ap = argparse.ArgumentParser(description="Qualitative per-method keyframe panel.")
    ap.add_argument("--episode", type=int, default=None,
                    help="global episode index to render (see --candidates)")
    ap.add_argument("--k", type=int, default=8, help="keyframe budget (default 8)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random extractor")
    ap.add_argument("--root", default="~/.cache/lerobot",
                    help="HuggingFace cache dir for the dataset")
    ap.add_argument("--out_dir", default="results/plots")
    ap.add_argument("--candidates", action="store_true",
                    help="list mid-length episodes (no decode) and exit")
    ap.add_argument("--min_len", type=int, default=30)
    ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--min_demos", type=int, default=20)
    args = ap.parse_args()

    from src.data.bridge_loader import BridgeDataLoader
    loader = BridgeDataLoader(root=args.root)

    if args.candidates or args.episode is None:
        rows = list_candidates(loader, args.min_len, args.max_len, args.min_demos)
        print(f"{len(rows)} mid-length episodes (T in [{args.min_len},{args.max_len}]); "
              f"pick a clear motion-then-settle clip:\n")
        print(f"{'episode':>8}  {'T':>4}  task")
        for ep, T, task in rows[:60]:
            print(f"{ep:>8}  {T:>4}  {task[:62]}")
        if args.episode is None:
            print("\nRe-run with --episode IDX to render the panel.")
            return

    demo = loader.load_episode(args.episode)
    frames = demo["images"]                       # (T, H, W, 3) uint8
    task = demo["task_name"]
    print(f"episode {args.episode}: T={len(frames)}, task='{task}'")

    picks = {name: np.asarray(ex.extract(frames), dtype=int)
             for name, ex in build_methods(args.k, args.seed)}
    for name, idx in picks.items():
        print(f"  {name:<13} -> {idx.tolist()}")

    out = render(frames, picks, task, args.k, args.out_dir)
    print(f"\nwrote {out} (+ .pdf)")
    print('\nSuggested caption: Representative episode '
          f'(T={len(frames)}, task "{task}"). Each row shows the frames selected by one '
          f'extractor at K={args.k}. Uniform spreads anchors evenly across the episode; '
          'optical_flow and attention concentrate them on low-motion "settled" frames, '
          "leaving temporal gaps -- which explains their worse coverage error "
          "(methods.md S5.6). Representative example, not selected to favour a result.")


if __name__ == "__main__":
    main()
