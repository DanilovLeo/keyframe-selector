"""
export_eval_bundle.py

Dump a single portable analysis bundle from the evaluation caches.

This runs *where the data lives* (the GPU pod): it reads the disk-cached CLIP
per-frame embeddings, re-derives the same gallery/query split and task selection
as the main eval, and (optionally) re-runs the extractor grid to capture each
method's keyframe indices.  The result is one self-contained bundle that every
downstream diagnostic (`scripts/diagnostics/*`) reads with **numpy alone** — no
GPU, no dataset download, no model load.  This is the bridge that turns the
audit's "cheap cached-embedding diagnostics" into actual local CPU work.

Why this exists
---------------
The per-frame CLIP embeddings live in `FrameEmbeddingCache` on the pod, not on
the analysis machine.  `eval_per_demo.jsonl` only stores scalar metrics, so none
of the Tier-1 diagnostics (intra-/inter-episode similarity, K=1, consecutive
block, oracle bound, bootstrap CIs) can be reconstructed locally without the
embeddings.  This script exports them once.

Bundle layout (default `results/bundle/`)
------------------------------------------
  frame_embeddings.npz   keys "ep{idx}" -> (T, D) float32 L2-normalised
  bundle_meta.json       config + per-episode {episode_index, task, task_id,
                         split, T}; plus the label list when indices are dumped
  keyframes.jsonl        one row per (episode, label):
                         {"episode_index", "label", "indices"}
                         (omitted with --no_indices)

The bundle is a derived artifact (~150-250 MB with embeddings); keep it local
and git-ignore `results/bundle/`.  Only the *diagnostic outputs* (small figs,
tables, JSON summaries) belong in version control.

Scope
-----
Pixels/embeddings in, indices out.  No policy, no rollout, no robot-state
signal, no new dataset — Variant C.

Usage (on the pod, from keyframe-selector/)
-------------------------------------------
    python scripts/export_eval_bundle.py \\
        --root        ~/.cache/lerobot         \\
        --embed_cache ~/.cache/kf_eval/clip_embeds \\
        --out_dir     results/bundle

    # embeddings only (no GPU / no image decode needed):
    python scripts/export_eval_bundle.py --no_indices
"""

from __future__ import annotations

# Match the eval scripts: cap CPU threads before numpy/torch spin up their
# OpenMP/MKL pools (avoids the many-vCPU oversubscription slowdown).
import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.cache import FrameEmbeddingCache

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"
_EXP_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "experiment.yaml"


def load_model_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_experiment_config() -> dict:
    with open(_EXP_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Grid constants — single source of truth: configs/experiment.yaml. The same
# file drives scripts/run_retrieval_eval.py, so the bundle reproduces the
# published configs exactly. METHODS is stamped into bundle_meta.json so the
# diagnostics read the grid back from the bundle (not the yaml).
_EXP = load_experiment_config()
K_SWEEP = _EXP["k_sweep"]
RANDOM_SEEDS = _EXP["random_seeds"]
METHODS = _EXP["methods"]
# Dataset sampling defaults (single source of truth: configs/experiment.yaml);
# the CLI flags below override these for ad-hoc runs.
_SAMPLING = _EXP.get("sampling", {})
MIN_DEMOS = _SAMPLING.get("min_demos", 20)
MAX_TASKS = _SAMPLING.get("max_tasks", 20)
MAX_EPISODES = _SAMPLING.get("max_episodes", 50)


# --------------------------------------------------------------------------- #
# Gallery/query split — copied verbatim from src/evaluation/retrieval.py for
# byte-identical parity with the main eval, without importing torch (so the
# embeddings-only path stays light).  Keep in sync if the original changes.
# --------------------------------------------------------------------------- #
def gallery_query_split(
    episode_indices: List[int],
    gallery_frac: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    arr = np.array(episode_indices, dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(arr)
    n_gallery = max(1, int(gallery_frac * len(arr)))
    n_gallery = min(n_gallery, len(arr) - 1)
    return arr[:n_gallery].tolist(), arr[n_gallery:].tolist()


def build_extractor_grid(cfg: dict) -> List[tuple]:
    """[(label, extractor), ...] — mirrors run_retrieval_eval.build_extractor_grid.

    Imported lazily: only needed when --dump_indices is on, so an
    embeddings-only export never pulls in torch / torchvision / timm.
    """
    from src.extractors import (  # noqa: PLC0415
        UniformExtractor,
        RandomExtractor,
        OpticalFlowExtractor,
        AttentionSaliencyExtractor,
        FrameDiffExtractor,
    )

    grid: List[tuple] = []
    for k in K_SWEEP:
        grid.append((f"uniform_k{k}", UniformExtractor(n_keyframes=k)))
    for k in K_SWEEP:
        for seed in RANDOM_SEEDS:
            grid.append((f"random_k{k}_s{seed}", RandomExtractor(n_keyframes=k, seed=seed)))
    for k in K_SWEEP:
        grid.append((f"optical_flow_k{k}", OpticalFlowExtractor(n_keyframes=k)))
    for k in K_SWEEP:
        grid.append((f"attention_k{k}", AttentionSaliencyExtractor(
            n_keyframes=k, timm_model=cfg["dinov2"]["timm_model"],
        )))
    for k in K_SWEEP:
        grid.append((f"frame_diff_k{k}", FrameDiffExtractor(n_keyframes=k)))
    return grid


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/.cache/lerobot",
                    help="HuggingFace/LeRobot cache root for BridgeDataLoader")
    ap.add_argument("--embed_cache", default=None,
                    help="Directory of disk-cached frame embeddings. Default: "
                         "~/.cache/kf_eval/<backbone>_embeds, a separate cache per "
                         "backbone so CLIP and DINOv2 embeddings never collide.")
    ap.add_argument("--backbone", choices=["clip", "dinov2"], default="clip",
                    help="Retrieval embedding backbone. 'clip' is the pinned primary "
                         "(ViT-L/14); 'dinov2' is the Task 4 vision-only cross-encoder "
                         "check (vit_small_patch14_dinov2). Selection extractors are "
                         "backbone-independent, so keyframes.jsonl is identical either "
                         "way; only frame_embeddings.npz changes.")
    ap.add_argument("--min_demos", type=int, default=MIN_DEMOS)
    ap.add_argument("--max_tasks", type=int, default=MAX_TASKS)
    ap.add_argument("--max_episodes", type=int, default=MAX_EPISODES)
    ap.add_argument("--out_dir", default="results/bundle")
    ap.add_argument("--no_indices", dest="dump_indices", action="store_false",
                    help="Skip the extractor grid; export embeddings + metadata only "
                         "(no image decode, no GPU).")
    ap.add_argument("--allow_embed", action="store_true",
                    help="On a cache miss, load CLIP and embed the episode "
                         "(needs GPU). Default: error out, since the point is to "
                         "reuse the existing cache.")
    args = ap.parse_args()

    cfg = load_model_config()
    clip_model_id = f"{cfg['clip']['model']}_{cfg['clip']['pretrained']}"
    dinov2_model_id = cfg["dinov2"]["timm_model"]
    model_id = clip_model_id if args.backbone == "clip" else dinov2_model_id
    embed_cache_dir = args.embed_cache or f"~/.cache/kf_eval/{args.backbone}_embeds"
    embed_cache = FrameEmbeddingCache(embed_cache_dir, model_id)
    print(f"Backbone        : {args.backbone}")
    print(f"Model id        : {model_id}")
    print(f"Embedding cache : {embed_cache}")
    print(f"Dump indices    : {args.dump_indices}")

    # ---- dataset index (metadata only) ------------------------------------ #
    from src.data.bridge_loader import BridgeDataLoader  # noqa: PLC0415
    loader = BridgeDataLoader(root=args.root)
    tasks = loader.list_tasks(min_demos=args.min_demos)[: args.max_tasks]
    if not tasks:
        sys.exit(f"No tasks with >= {args.min_demos} demos found.")
    print(f"Tasks           : {len(tasks)}  (max_episodes={args.max_episodes})")

    # ---- optional extractor grid + lazy CLIP for embed-on-miss ------------ #
    grid = build_extractor_grid(cfg) if args.dump_indices else []
    if args.dump_indices:
        print(f"Extractor grid  : {len(grid)} configs")

    encoder = None  # (model, preprocess, device, embed_fn) — lazily loaded on miss

    def embed_on_miss(images: np.ndarray) -> np.ndarray:
        nonlocal encoder
        if not args.allow_embed:
            sys.exit("Cache miss and --allow_embed not set; aborting so we never "
                     "silently recompute embeddings with a different setup.")
        if encoder is None:
            if args.backbone == "clip":
                from src.evaluation.retrieval import (  # noqa: PLC0415
                    load_clip, embed_all_frames)
                m, pre, _tok, dev = load_clip(model_name=cfg["clip"]["model"],
                                              pretrained=cfg["clip"]["pretrained"])
                encoder = (m, pre, dev, embed_all_frames)
            else:
                from src.evaluation.retrieval import (  # noqa: PLC0415
                    load_dinov2, embed_all_frames_dinov2)
                m, pre, dev = load_dinov2(timm_model=dinov2_model_id)
                encoder = (m, pre, dev, embed_all_frames_dinov2)
        m, pre, dev, embed_fn = encoder
        return embed_fn(images, m, pre, dev)

    # ---- main loop -------------------------------------------------------- #
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_arrays: Dict[str, np.ndarray] = {}
    ep_meta: List[dict] = []
    kf_rows: List[dict] = []
    n_missing = 0

    for task_id, task in enumerate(tasks):
        ep_indices = loader.list_episodes(task)[: args.max_episodes]
        gallery_eps, query_eps = gallery_query_split(ep_indices, seed=42)
        split_of = {e: "gallery" for e in gallery_eps}
        split_of.update({e: "query" for e in query_eps})

        # gallery first then query, mirroring the eval's record order
        ordered = gallery_eps + query_eps
        print(f"[{task_id + 1:>2}/{len(tasks)}] {task[:60]:<60} "
              f"({len(ordered)} eps: {len(gallery_eps)}g/{len(query_eps)}q)")

        for ep in ordered:
            fe = embed_cache.get(ep)
            need_images = args.dump_indices or fe is None

            images: Optional[np.ndarray] = None
            if need_images:
                images = loader.load_episode(ep)["images"]

            if fe is None:
                fe = embed_on_miss(images)
                embed_cache.put(ep, fe)
                n_missing += 1

            fe = np.asarray(fe, dtype=np.float32)
            T = int(fe.shape[0])
            emb_arrays[f"ep{ep}"] = fe
            ep_meta.append({
                "episode_index": int(ep),
                "task": task,
                "task_id": int(task_id),
                "split": split_of[ep],
                "T": T,
            })

            if args.dump_indices:
                for label, extractor in grid:
                    idx = np.asarray(extractor.extract(images)).astype(int).tolist()
                    kf_rows.append({
                        "episode_index": int(ep),
                        "label": label,
                        "indices": idx,
                    })

    # ---- write bundle ----------------------------------------------------- #
    emb_path = out_dir / "frame_embeddings.npz"
    np.savez(emb_path, **emb_arrays)
    print(f"\nframe_embeddings.npz -> {emb_path}  ({len(emb_arrays)} episodes)")

    meta = {
        "config": {
            "tasks": tasks,
            "min_demos": args.min_demos,
            "max_tasks": args.max_tasks,
            "max_episodes": args.max_episodes,
            "retrieval_backbone": args.backbone,
            "embedding_model": model_id,
            "clip_model": clip_model_id,
            "dinov2_model": dinov2_model_id,
            "random_seeds": RANDOM_SEEDS,
            "K_sweep": K_SWEEP,
            "methods": METHODS,
            "embedding_dtype": "float32",
            "has_indices": args.dump_indices,
        },
        "labels": [label for label, _ in grid],
        "episodes": ep_meta,
    }
    meta_path = out_dir / "bundle_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"bundle_meta.json     -> {meta_path}  ({len(ep_meta)} episode records)")

    if args.dump_indices:
        kf_path = out_dir / "keyframes.jsonl"
        with open(kf_path, "w") as f:
            for row in kf_rows:
                f.write(json.dumps(row) + "\n")
        print(f"keyframes.jsonl      -> {kf_path}  ({len(kf_rows)} rows)")

    if n_missing:
        print(f"\nNOTE: {n_missing} episode(s) were embedded on the fly (cache miss).")
    print("\nDone. Copy results/bundle/ to the analysis machine for diagnostics.")


if __name__ == "__main__":
    main()
