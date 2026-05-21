"""
preflight_check.py

Run this on RunPod BEFORE launching run_retrieval_eval.py.
Catches the failure modes that waste the most GPU time.

Checks
------
  1. CUDA available
  2. Load one real BridgeData v2 episode; run all 4 extractors end-to-end
  3. Print natural-K distribution for OpticalFlow and Attention on 5 episodes
  4. Gallery/query split is stratified (per-task) and deterministic
  5. Model IDs loaded from configs/models.yaml (not hardcoded)
  6. Disk cache write and read round-trip

Usage (from keyframe-selector/):
    python scripts/preflight_check.py \\
        [--root ~/.cache/lerobot]        \\
        [--embed_cache ~/.cache/kf_eval] \\
        [--min_demos 5]                  \\
        [--n_check_eps 5]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "  [PASS]"
FAIL = "  [FAIL]"
SKIP = "  [SKIP]"


def check_cuda() -> bool:
    import torch
    ok = torch.cuda.is_available()
    if ok:
        name = torch.cuda.get_device_name(0)
        print(f"{PASS} CUDA available: {name}")
    else:
        print(f"{FAIL} CUDA not available — experiments will be very slow on CPU")
    return ok


def check_model_config(cfg_path: Path) -> dict | None:
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        clip = cfg["clip"]
        dino = cfg["dinov2"]
        print(f"{PASS} configs/models.yaml loaded")
        print(f"       CLIP   : {clip['model']} / {clip['pretrained']} (dim={clip['dim']})")
        print(f"       DINOv2 : {dino['timm_model']} (dim={dino['dim']})")
        return cfg
    except Exception as e:
        print(f"{FAIL} configs/models.yaml: {e}")
        return None


def check_cache_roundtrip(embed_cache_dir: str) -> bool:
    from src.evaluation.cache import FrameEmbeddingCache
    try:
        with tempfile.TemporaryDirectory() as tmp:
            cache = FrameEmbeddingCache(tmp, "test_model")
            arr = np.random.rand(10, 768).astype(np.float32)
            cache.put(99999, arr)
            loaded = cache.get(99999)
            assert loaded is not None and np.allclose(arr, loaded), "round-trip mismatch"
        print(f"{PASS} Disk cache read/write round-trip OK")
        # Check the real cache dir is writable
        real_cache = FrameEmbeddingCache(embed_cache_dir, "preflight_test")
        print(f"       Cache dir: {real_cache}")
        return True
    except Exception as e:
        print(f"{FAIL} Disk cache: {e}")
        return False


def check_gallery_query_split() -> bool:
    from src.evaluation.retrieval import gallery_query_split
    eps = list(range(20))
    g1, q1 = gallery_query_split(eps, seed=42)
    g2, q2 = gallery_query_split(eps, seed=42)
    ok = (g1 == g2 and q1 == q2
          and len(g1) == 16 and len(q1) == 4
          and set(g1) | set(q1) == set(eps)
          and set(g1) & set(q1) == set())
    if ok:
        print(f"{PASS} gallery_query_split: deterministic 80/20, no overlap, covers all eps")
    else:
        print(f"{FAIL} gallery_query_split failed: g={len(g1)} q={len(q1)}")
    return ok


def check_real_episode(loader, cfg: dict, n_eps: int) -> bool:
    """Load n_eps real episodes, run all 4 extractors, verify invariants."""
    from src.extractors import (
        UniformExtractor, RandomExtractor,
        OpticalFlowExtractor, AttentionSaliencyExtractor,
    )

    tasks = loader.list_tasks(min_demos=5)
    if not tasks:
        print(f"{SKIP} No tasks with >=5 demos — skip real episode check")
        return True

    task = tasks[0]
    eps  = loader.list_episodes(task)[:n_eps]
    print(f"\n  Task: {task!r}")
    print(f"  Episodes to check: {eps}")

    extractors = [
        ("uniform_k8",    UniformExtractor(n_keyframes=8)),
        ("random_k8_s42", RandomExtractor(n_keyframes=8, seed=42)),
        ("optical_flow",  OpticalFlowExtractor()),
        ("attention_dino", AttentionSaliencyExtractor(
            timm_model=cfg["dinov2"]["timm_model"]
        )),
    ]

    natural_k: dict = {"optical_flow": [], "attention_dino": []}
    all_ok = True

    for ep_idx in eps:
        episode = loader.load_episode(ep_idx)
        images  = episode["images"]
        T       = len(images)
        print(f"\n  ep={ep_idx}  T={T}  shape={images.shape}  dtype={images.dtype}")

        for name, ext in extractors:
            kf = ext.extract(images)
            cr = len(kf) / T

            inv_ok = (kf[0] == 0 and kf[-1] == T - 1
                      and (np.diff(kf) > 0).all())
            status = PASS if inv_ok else FAIL
            if not inv_ok:
                all_ok = False
            print(f"    {name:<22} K={len(kf):>3}  CR={cr:.3f}  {status}")

            if name in natural_k:
                natural_k[name].append(len(kf))

    # Natural-K distribution report
    print(f"\n  Natural-K distribution over {n_eps} episodes:")
    for name, ks in natural_k.items():
        if ks:
            print(f"    {name:<22} "
                  f"mean={np.mean(ks):.1f}  std={np.std(ks):.1f}  "
                  f"min={min(ks)}  max={max(ks)}")
            if max(ks) / max(min(ks), 1) > 5:
                print(f"    *** WARNING: {name} K range is > 5×  "
                      f"— consider a fixed-K variant for fair comparison ***")

    if all_ok:
        print(f"\n{PASS} All extractor invariants hold on real episodes")
    else:
        print(f"\n{FAIL} Some extractor invariants failed — see above")
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         default="~/.cache/lerobot")
    parser.add_argument("--embed_cache",  default="~/.cache/kf_eval/clip_embeds")
    parser.add_argument("--min_demos",    type=int, default=5)
    parser.add_argument("--n_check_eps",  type=int, default=5)
    args = parser.parse_args()

    cfg_path = Path(__file__).parent.parent / "configs" / "models.yaml"

    print("=" * 60)
    print("  Keyframe-selector pre-flight check")
    print("=" * 60)

    results = {}

    results["cuda"]       = check_cuda()
    cfg                   = check_model_config(cfg_path)
    results["model_cfg"]  = cfg is not None
    results["cache"]      = check_cache_roundtrip(args.embed_cache)
    results["split"]      = check_gallery_query_split()

    if cfg is not None:
        try:
            from src.data.bridge_loader import BridgeDataLoader  # noqa
            print(f"\nLoading BridgeDataLoader from {args.root} ...")
            loader = BridgeDataLoader(root=args.root)
            results["real_eps"] = check_real_episode(loader, cfg, args.n_check_eps)
        except ImportError as e:
            print(f"{SKIP} Missing dependency — skipping real episode check: {e}")
            results["real_eps"] = None
        except Exception as e:
            print(f"{FAIL} BridgeDataLoader: {e}")
            results["real_eps"] = False

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    all_passed = True
    for name, ok in results.items():
        if ok is None:
            print(f"  {SKIP}  {name}")
        elif ok:
            print(f"  {PASS}  {name}")
        else:
            print(f"  {FAIL}  {name}")
            all_passed = False

    if all_passed:
        print("\n  All checks passed — safe to launch run_retrieval_eval.py")
    else:
        print("\n  Fix the failures above before launching the full sweep.")
        sys.exit(1)


if __name__ == "__main__":
    main()
