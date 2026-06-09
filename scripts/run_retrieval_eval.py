"""
run_retrieval_eval.py

Full evaluation of all keyframe extractors on BridgeData v2.

For each task (with >= min_demos episodes):
  - Shuffle episodes seed=42, split 80/20 per-task (stratified).
  - Embed ALL frames of every episode with CLIP ViT-L/14 — done once,
    cached to disk keyed by (episode_index, model_id).  Re-runs are fast.
  - For each (extractor, K) config, pool keyframe embeddings → demo vector.
  - Aggregate all tasks into a multi-class retrieval problem and compute:
      · Top-1 and Top-5 retrieval accuracy
      · Mean CLIP text-image cosine similarity
      · Mean compression ratio (actual keyframes / T)
  - Random baseline uses 3 seeds; results reported as mean ± std.

Extractor grid
--------------
  Uniform: K ∈ {4, 8, 16, 32}  (exact)
  Random:  K ∈ {4, 8, 16, 32}  × seeds {42, 123, 456}
  OpticalFlow, AttentionSaliency: natural K (default parameters)

Outputs
-------
  results/eval_retrieval.json       — aggregated metrics per extractor
  results/eval_per_demo.jsonl       — one JSON line per demo per extractor
                                       (use for sub-group analysis later)

Usage
-----
    python scripts/run_retrieval_eval.py \\
        [--root ~/.cache/lerobot]         \\
        [--embed_cache ~/.cache/kf_eval]  \\
        [--min_demos 20]                  \\
        [--max_tasks 20]                  \\
        [--max_episodes 50]               \\
        [--output_dir results]
"""

from __future__ import annotations

# Cap CPU threads BEFORE numpy/torch initialise their OpenMP/MKL runtimes.
# On many-vCPU hosts (e.g. RunPod A100 pods) torch otherwise opens a parallel
# region across every core for each small op; the thread launch/sync overhead
# makes ViT forwards ~80x slower (measured: 41s -> 0.5s per episode). setdefault
# means an explicit `export OMP_NUM_THREADS=...` still overrides this.
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors import (
    UniformExtractor,
    RandomExtractor,
    OpticalFlowExtractor,
    AttentionSaliencyExtractor,
    FrameDiffExtractor,
)
from src.evaluation.retrieval import (
    load_clip,
    embed_all_frames,
    pool_demo_embedding,
    gallery_query_split,
    retrieval_accuracy,
)
from src.evaluation.clip_similarity import mean_clip_similarity, embed_text, cosine_sim
from src.evaluation.cache import FrameEmbeddingCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"


def load_model_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

K_SWEEP      = [4, 8, 16, 32]
RANDOM_SEEDS = [42, 123, 456]


def build_extractor_grid(cfg: dict) -> List[tuple]:
    """Return [(label, extractor), ...] for the full experiment grid."""
    grid = []
    for k in K_SWEEP:
        grid.append((f"uniform_k{k}", UniformExtractor(n_keyframes=k)))
    for k in K_SWEEP:
        for seed in RANDOM_SEEDS:
            grid.append((f"random_k{k}_s{seed}", RandomExtractor(n_keyframes=k, seed=seed)))
    for k in K_SWEEP:
        grid.append((f"optical_flow_k{k}", OpticalFlowExtractor(n_keyframes=k)))
    for k in K_SWEEP:
        grid.append((f"attention_k{k}", AttentionSaliencyExtractor(
            n_keyframes=k, timm_model=cfg["dinov2"]["timm_model"]
        )))
    for k in K_SWEEP:
        grid.append((f"frame_diff_k{k}", FrameDiffExtractor(n_keyframes=k)))
    return grid


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def eval_task(
    loader,
    task: str,
    task_id: int,
    extractor_grid: list,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    clip_device,
    embed_cache: FrameEmbeddingCache,
    max_episodes: int,
) -> List[dict]:
    """Evaluate all extractors on one task.

    Returns a flat list of per-demo records (one per extractor × episode).
    Each record is serialisable (no numpy arrays).
    """
    ep_indices = loader.list_episodes(task)[:max_episodes]
    gallery_eps, query_eps = gallery_query_split(ep_indices, seed=42)

    # Pre-compute text embedding for CLIP similarity
    text_emb = embed_text(task, clip_model, clip_tokenizer, clip_device)

    # Image cache: {ep_idx: (images, frame_embs)}
    img_cache: Dict[int, tuple] = {}

    def get_episode(ep_idx: int):
        if ep_idx not in img_cache:
            episode = loader.load_episode(ep_idx)
            images = episode["images"]
            # CLIP frame embeddings: disk cache first
            fe = embed_cache.get(ep_idx)
            if fe is None:
                fe = embed_all_frames(images, clip_model, clip_preprocess, clip_device)
                embed_cache.put(ep_idx, fe)
            img_cache[ep_idx] = (images, fe)
        return img_cache[ep_idx]

    records: List[dict] = []

    for split_name, split_eps in [("gallery", gallery_eps), ("query", query_eps)]:
        for ep_idx in split_eps:
            images, frame_embs = get_episode(ep_idx)
            T = len(images)

            for label, extractor in extractor_grid:
                kf  = extractor.extract(images)
                cr  = len(kf) / T if T > 0 else 0.0
                dem = pool_demo_embedding(frame_embs, kf)
                sim = cosine_sim(dem, text_emb)

                records.append({
                    "extractor":      label,
                    "task":           task,
                    "task_id":        task_id,
                    "episode_index":  ep_idx,
                    "split":          split_name,
                    "n_kf":           int(len(kf)),
                    "cr":             float(cr),
                    "T":              int(T),
                    "clip_sim":       float(sim),
                    # embedding stored separately in cache; not logged here
                    "_embedding":     dem,   # used for retrieval; stripped before saving
                })

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(
    all_records: List[dict],
    extractor_grid: list,
) -> Dict[str, dict]:
    """Compute retrieval accuracy and mean metrics per extractor label."""
    labels = [label for label, _ in extractor_grid]
    results: Dict[str, dict] = {}

    for label in labels:
        recs = [r for r in all_records if r["extractor"] == label]
        gallery = [r for r in recs if r["split"] == "gallery"]
        query   = [r for r in recs if r["split"] == "query"]
        if not gallery or not query:
            continue

        g_embs   = np.stack([r["_embedding"] for r in gallery])
        g_labels = np.array([r["task_id"]    for r in gallery])
        q_embs   = np.stack([r["_embedding"] for r in query])
        q_labels = np.array([r["task_id"]    for r in query])

        acc = retrieval_accuracy(g_embs, g_labels, q_embs, q_labels, top_k=(1, 5))

        results[label] = {
            **acc,
            "mean_cr":    float(np.mean([r["cr"]       for r in recs])),
            "mean_n_kf":  float(np.mean([r["n_kf"]     for r in recs])),
            "clip_sim":   float(np.mean([r["clip_sim"]  for r in recs])),
            "n_gallery":  len(gallery),
            "n_query":    len(query),
        }

    # Add mean ± std entries for Random (averaged across seeds)
    for k in K_SWEEP:
        seed_keys = [f"random_k{k}_s{s}" for s in RANDOM_SEEDS]
        seed_res  = [results[key] for key in seed_keys if key in results]
        if not seed_res:
            continue
        results[f"random_k{k}"] = {
            "top_1":      float(np.mean([r["top_1"]     for r in seed_res])),
            "top_1_std":  float(np.std( [r["top_1"]     for r in seed_res])),
            "top_5":      float(np.mean([r["top_5"]     for r in seed_res])),
            "top_5_std":  float(np.std( [r["top_5"]     for r in seed_res])),
            "clip_sim":   float(np.mean([r["clip_sim"]  for r in seed_res])),
            "clip_sim_std": float(np.std([r["clip_sim"] for r in seed_res])),
            "mean_cr":    float(np.mean([r["mean_cr"]   for r in seed_res])),
            "mean_n_kf":  float(np.mean([r["mean_n_kf"] for r in seed_res])),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         default="~/.cache/lerobot")
    parser.add_argument("--embed_cache",  default="~/.cache/kf_eval/clip_embeds",
                        help="Directory for disk-cached CLIP frame embeddings")
    parser.add_argument("--min_demos",    type=int, default=20)
    parser.add_argument("--max_tasks",    type=int, default=20)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--output_dir",   default="results")
    args = parser.parse_args()

    cfg = load_model_config()
    print(f"Models config: CLIP={cfg['clip']['model']}/{cfg['clip']['pretrained']}  "
          f"DINOv2={cfg['dinov2']['timm_model']}")

    from src.data.bridge_loader import BridgeDataLoader  # noqa: PLC0415

    print("Loading BridgeDataLoader ...")
    loader = BridgeDataLoader(root=args.root)
    tasks  = loader.list_tasks(min_demos=args.min_demos)[: args.max_tasks]
    if not tasks:
        sys.exit(f"No tasks with >= {args.min_demos} demos found.")
    print(f"Tasks: {len(tasks)}   max_episodes={args.max_episodes}")

    print("Loading CLIP ViT-L/14 ...")
    clip_model, clip_preprocess, clip_tokenizer, clip_device = load_clip(
        model_name=cfg["clip"]["model"],
        pretrained=cfg["clip"]["pretrained"],
    )
    print(f"  device: {clip_device}")

    model_id     = f"{cfg['clip']['model']}_{cfg['clip']['pretrained']}"
    embed_cache  = FrameEmbeddingCache(args.embed_cache, model_id)
    print(f"  embedding cache: {embed_cache}")

    extractor_grid = build_extractor_grid(cfg)
    print(f"  extractor configs: {len(extractor_grid)}  "
          f"({len(K_SWEEP)} uniform + {len(K_SWEEP)*len(RANDOM_SEEDS)} random "
          f"+ {3*len(K_SWEEP)} CV)\n")

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    all_records: List[dict] = []

    for task_id, task in enumerate(tasks):
        ep_count = loader.num_episodes_for(task)
        print(f"[{task_id+1:>2}/{len(tasks)}] {task[:65]:<65} ({ep_count} eps)")

        task_records = eval_task(
            loader, task, task_id, extractor_grid,
            clip_model, clip_preprocess, clip_tokenizer, clip_device,
            embed_cache, args.max_episodes,
        )
        all_records.extend(task_records)

    # ------------------------------------------------------------------ #
    # Aggregate                                                            #
    # ------------------------------------------------------------------ #
    print("\nAggregating ...")
    agg = aggregate(all_records, extractor_grid)

    # ------------------------------------------------------------------ #
    # Print summary                                                        #
    # ------------------------------------------------------------------ #
    HDR = f"{'Extractor':<24} {'Top-1':>6} {'±':>4} {'Top-5':>6} {'±':>4} {'CLIP-sim':>9} {'mean_CR':>8} {'mean_KF':>8}"
    SEP = "-" * len(HDR)
    print(f"\n{SEP}\n{HDR}\n{SEP}")

    # Print in a logical order: uniform, random (averaged), then CV methods
    display_order = (
        [f"uniform_k{k}" for k in K_SWEEP]
        + [f"random_k{k}" for k in K_SWEEP]
        + [f"optical_flow_k{k}" for k in K_SWEEP]
        + [f"attention_k{k}" for k in K_SWEEP]
        + [f"frame_diff_k{k}" for k in K_SWEEP]
    )
    for label in display_order:
        res = agg.get(label)
        if res is None:
            continue
        t1_std = res.get("top_1_std", 0.0)
        t5_std = res.get("top_5_std", 0.0)
        print(
            f"{label:<24} {res['top_1']:>6.3f} {t1_std:>4.3f} "
            f"{res['top_5']:>6.3f} {t5_std:>4.3f} "
            f"{res['clip_sim']:>9.4f} {res['mean_cr']:>8.4f} {res['mean_n_kf']:>8.1f}"
        )

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregated results
    agg_path = out_dir / "eval_retrieval.json"
    with open(agg_path, "w") as f:
        json.dump({
            "config": {
                "tasks":        tasks,
                "min_demos":    args.min_demos,
                "max_tasks":    args.max_tasks,
                "max_episodes": args.max_episodes,
                "clip_model":   model_id,
                "dinov2_model": cfg["dinov2"]["timm_model"],
                "random_seeds": RANDOM_SEEDS,
                "K_sweep":      K_SWEEP,
            },
            "results": agg,
        }, f, indent=2)
    print(f"\nAggregated  → {agg_path}")

    # Per-demo records (strip _embedding before writing)
    perdemo_path = out_dir / "eval_per_demo.jsonl"
    with open(perdemo_path, "w") as f:
        for rec in all_records:
            row = {k: v for k, v in rec.items() if k != "_embedding"}
            f.write(json.dumps(row) + "\n")
    print(f"Per-demo    → {perdemo_path}  ({len(all_records)} records)")


if __name__ == "__main__":
    main()
