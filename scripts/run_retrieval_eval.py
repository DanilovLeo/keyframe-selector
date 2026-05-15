"""
run_retrieval_eval.py

Full evaluation of all keyframe extractors on BridgeData v2.

For each task (with >= min_demos episodes):
  - Shuffle episodes with seed=42, split 80/20 into gallery / query.
  - Embed ALL frames of every episode with CLIP ViT-L/14 (done once per episode,
    reused across all extractors and K values).
  - For each (extractor, K) configuration, pool keyframe embeddings → demo vector.
  - Aggregate all tasks into a multi-class retrieval problem and compute:
      · Top-1 and Top-5 retrieval accuracy
      · Mean CLIP text-image cosine similarity
      · Mean compression ratio (actual keyframes / T)

Extractor grid
--------------
  Uniform and Random are swept over K ∈ {4, 8, 16, 32} (exact keyframe counts).
  OpticalFlow and AttentionSaliency use their natural keyframe counts (one entry
  per method with their default parameters).

Usage
-----
    python scripts/run_retrieval_eval.py \\
        [--root ~/.cache/lerobot]        \\
        [--min_demos 20]                 \\
        [--max_tasks 20]                 \\
        [--max_episodes 50]              \\
        [--output results/eval.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors import (
    UniformExtractor,
    RandomExtractor,
    OpticalFlowExtractor,
    AttentionSaliencyExtractor,
)
from src.evaluation.retrieval import (
    load_clip,
    embed_all_frames,
    pool_demo_embedding,
    gallery_query_split,
    retrieval_accuracy,
)
from src.evaluation.clip_similarity import mean_clip_similarity

# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

K_SWEEP = [4, 8, 16, 32]

def build_extractor_grid():
    """Return list of (label, extractor) for the full experiment grid."""
    grid = []
    for k in K_SWEEP:
        grid.append((f"uniform_k{k}",  UniformExtractor(n_keyframes=k)))
        grid.append((f"random_k{k}",   RandomExtractor(n_keyframes=k, seed=42)))
    grid.append(("optical_flow",    OpticalFlowExtractor()))
    grid.append(("attention_dino",  AttentionSaliencyExtractor()))
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
    max_episodes: int,
    verbose: bool = True,
) -> dict:
    """Compute gallery / query embeddings for one task.

    Returns a dict mapping extractor_label → list of
    {"split": "gallery"|"query", "task_id": int, "embedding": ndarray, "cr": float}
    records, plus the task string and episode count.
    """
    ep_indices = loader.list_episodes(task)[:max_episodes]
    gallery_eps, query_eps = gallery_query_split(ep_indices)

    # Cache per-episode: {ep_idx: (images (T,H,W,3), frame_embs (T,D))}
    ep_cache: Dict[int, tuple] = {}

    def get_episode(ep_idx: int):
        if ep_idx not in ep_cache:
            episode = loader.load_episode(ep_idx)
            images = episode["images"]
            fe = embed_all_frames(images, clip_model, clip_preprocess, clip_device)
            ep_cache[ep_idx] = (images, fe)
        return ep_cache[ep_idx]

    records: Dict[str, List[dict]] = {label: [] for label, _ in extractor_grid}

    for split_name, split_eps in [("gallery", gallery_eps), ("query", query_eps)]:
        for ep_idx in split_eps:
            episode_images, frame_embs = get_episode(ep_idx)
            T = len(episode_images)

            for label, extractor in extractor_grid:
                kf = extractor.extract(episode_images)
                cr = len(kf) / T if T > 0 else 0.0
                demo_emb = pool_demo_embedding(frame_embs, kf)
                records[label].append({
                    "split":     split_name,
                    "task_id":   task_id,
                    "embedding": demo_emb,
                    "cr":        cr,
                    "n_kf":      len(kf),
                })

    return records


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_retrieval(
    all_records: Dict[str, List[dict]],
    extractor_grid: list,
) -> Dict[str, dict]:
    """Run retrieval evaluation over the pooled multi-task gallery / query sets.

    Returns a dict: extractor_label → {top_1, top_5, mean_cr, mean_kf}.
    """
    results = {}
    for label, _ in extractor_grid:
        recs = all_records[label]

        gallery = [r for r in recs if r["split"] == "gallery"]
        query   = [r for r in recs if r["split"] == "query"]

        if not gallery or not query:
            continue

        g_embs   = np.stack([r["embedding"] for r in gallery])
        g_labels = np.array([r["task_id"]   for r in gallery])
        q_embs   = np.stack([r["embedding"] for r in query])
        q_labels = np.array([r["task_id"]   for r in query])

        acc = retrieval_accuracy(g_embs, g_labels, q_embs, q_labels, top_k=(1, 5))

        all_crs = [r["cr"] for r in recs]
        all_kfs = [r["n_kf"] for r in recs]

        results[label] = {
            **acc,
            "mean_cr":  float(np.mean(all_crs)),
            "mean_n_kf": float(np.mean(all_kfs)),
            "n_gallery": len(gallery),
            "n_query":   len(query),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         default="~/.cache/lerobot")
    parser.add_argument("--min_demos",    type=int, default=20)
    parser.add_argument("--max_tasks",    type=int, default=20)
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--output",       default="results/eval_retrieval.json")
    args = parser.parse_args()

    from src.data.bridge_loader import BridgeDataLoader  # noqa: PLC0415

    print("Loading BridgeDataLoader ...")
    loader = BridgeDataLoader(root=args.root)
    tasks  = loader.list_tasks(min_demos=args.min_demos)[: args.max_tasks]
    if not tasks:
        sys.exit(f"No tasks with >= {args.min_demos} demos found.")
    print(f"Tasks: {len(tasks)}   Episodes capped at {args.max_episodes} per task")

    print("Loading CLIP ViT-L/14 ...")
    clip_model, clip_preprocess, clip_tokenizer, clip_device = load_clip()
    print(f"  device: {clip_device}")

    extractor_grid = build_extractor_grid()
    print(f"Extractor grid: {[l for l,_ in extractor_grid]}\n")

    # Collect records across all tasks
    all_records: Dict[str, List[dict]]  = {label: [] for label, _ in extractor_grid}
    clip_sims:   Dict[str, List[float]] = {label: [] for label, _ in extractor_grid}

    for task_id, task in enumerate(tasks):
        short = task[:60]
        ep_count = loader.num_episodes_for(task)
        print(f"[{task_id+1:>2}/{len(tasks)}] {short:<60} ({ep_count} eps)")

        task_records = eval_task(
            loader, task, task_id, extractor_grid,
            clip_model, clip_preprocess, clip_tokenizer, clip_device,
            max_episodes=args.max_episodes,
        )

        for label in all_records:
            recs = task_records[label]
            all_records[label].extend(recs)

            # CLIP similarity: all episodes for this task under this extractor
            demo_embs = [r["embedding"] for r in recs]
            sim = mean_clip_similarity(demo_embs, task, clip_model, clip_tokenizer, clip_device)
            clip_sims[label].append(sim)

    # Aggregate
    print("\nAggregating retrieval metrics ...")
    retrieval_results = aggregate_retrieval(all_records, extractor_grid)

    # Merge CLIP similarity into results
    for label in retrieval_results:
        retrieval_results[label]["clip_sim"] = float(np.mean(clip_sims[label]))

    # Print summary table
    HDR = f"{'Extractor':<22} {'Top-1':>6} {'Top-5':>6} {'CLIP-sim':>9} {'mean_CR':>8} {'mean_KF':>8}"
    SEP = "-" * len(HDR)
    print(f"\n{SEP}\n{HDR}\n{SEP}")
    for label, res in retrieval_results.items():
        print(
            f"{label:<22} {res.get('top_1', 0):>6.3f} {res.get('top_5', 0):>6.3f} "
            f"{res.get('clip_sim', 0):>9.4f} {res.get('mean_cr', 0):>8.4f} "
            f"{res.get('mean_n_kf', 0):>8.1f}"
        )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert embeddings are not serialisable — strip them before saving
    serialisable = {}
    for label, res in retrieval_results.items():
        serialisable[label] = {k: v for k, v in res.items()}

    payload = {
        "config": {
            "tasks":        tasks,
            "min_demos":    args.min_demos,
            "max_tasks":    args.max_tasks,
            "max_episodes": args.max_episodes,
            "clip_model":   "ViT-L-14/openai",
        },
        "results": serialisable,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
