"""
smoke_test.py

Quick end-to-end sanity check.  Runs all available extractors on one
BridgeData v2 episode and verifies the invariants that every extractor
must satisfy:
  - returned indices are sorted ascending
  - first index == 0
  - last index  == T - 1

Usage (from keyframe-selector/):
    python smoke_test.py                         # uses real BridgeData v2 episode
    python smoke_test.py --synthetic             # uses a synthetic 120-frame episode
    python smoke_test.py --task "pick up..."     # override task string
    python smoke_test.py --episode 5             # override episode index

BridgeData v2 data is downloaded via lerobot on first run (~few GB for the
metadata; individual episode images are fetched lazily).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.extractors import KeyframeExtractor, UniformExtractor, RandomExtractor

# Collect available extractors; image-based ones added once Phase 3/4 land
EXTRACTORS: list[tuple[str, KeyframeExtractor]] = [
    ("uniform_10", UniformExtractor(n_keyframes=10)),
    ("random_10",  RandomExtractor(n_keyframes=10, seed=42)),
]

try:
    from src.extractors.optical_flow import OpticalFlowExtractor
    EXTRACTORS.append(("optical_flow", OpticalFlowExtractor()))
except ImportError:
    pass

try:
    from src.extractors.attention import AttentionSaliencyExtractor
    EXTRACTORS.append(("attention_dino", AttentionSaliencyExtractor()))
except ImportError:
    pass


def make_synthetic_episode(T: int = 120, H: int = 256, W: int = 256) -> dict:
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
    return {"images": images, "task_name": "synthetic: pick up the block", "n_frames": T}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a synthetic episode instead of downloading real data")
    parser.add_argument("--task",    default="",
                        help="Task substring to select (default: first task with >= 20 demos)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index within the selected task (default: 0)")
    parser.add_argument("--root",    default="~/.cache/lerobot",
                        help="Local lerobot dataset cache root")
    args = parser.parse_args()

    if args.synthetic:
        episode = make_synthetic_episode()
        print(f"[synthetic]  T={episode['n_frames']}  task={episode['task_name']!r}")
    else:
        from src.data.bridge_loader import BridgeDataLoader

        print(f"Loading BridgeDataLoader from {args.root} ...")
        loader = BridgeDataLoader(root=args.root)

        if args.task:
            candidates = [t for t in loader.list_tasks(min_demos=1) if args.task in t]
            if not candidates:
                sys.exit(f"No task matching {args.task!r}. "
                         f"Try: {loader.list_tasks(min_demos=20)[:5]}")
            task = candidates[0]
        else:
            tasks = loader.list_tasks(min_demos=20)
            if not tasks:
                sys.exit("No tasks with >= 20 demos found. Check the dataset.")
            task = tasks[0]

        eps = loader.list_episodes(task)
        ep_idx = eps[args.episode]
        print(f"Task  : {task!r}")
        print(f"Episode index: {ep_idx}  ({len(eps)} episodes for this task)")

        episode = loader.load_episode(ep_idx)

    images = episode["images"]
    T = episode["n_frames"]
    print(f"Loaded {T} frames  shape={images.shape}  dtype={images.dtype}\n")

    print(f"{'Method':<18} {'N':>4} {'CR':>7}  Indices (first 8)")
    print("-" * 72)
    all_passed = True
    for name, ext in EXTRACTORS:
        kf = ext.extract(images)
        cr = KeyframeExtractor.compression_ratio(T, len(kf))
        snippet = kf[:8].tolist()
        print(f"{name:<18} {len(kf):>4} {cr:>7.3f}  {snippet}")

        ok = True
        if kf[0] != 0:
            print(f"  FAIL: first index {kf[0]} != 0");  ok = False
        if kf[-1] != T - 1:
            print(f"  FAIL: last index {kf[-1]} != {T-1}");  ok = False
        if not (np.diff(kf) > 0).all():
            print(f"  FAIL: indices not strictly ascending");  ok = False
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("All sanity checks passed.")
    else:
        print("SOME CHECKS FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
