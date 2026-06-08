"""
run_consistency.py

Keyframe consistency check over BridgeData v2.

For each (task, extractor) pair reports:
  mean_kf / std_kf  — average and std of keyframe count across episodes
  cv_kf             — coefficient of variation (std/mean); lower = more consistent
  mean_cr           — mean compression ratio (n_keyframes / T)

Extractors are discovered automatically — Uniform and Random are always
available; OpticalFlow and AttentionSaliency are included once Phase 3/4
are written.

Output:
  stdout             — formatted table
  results/consistency_check_bridge.json — full results

Usage (from keyframe-selector/):
    python scripts/run_consistency.py \\
        [--root ~/.cache/lerobot] \\
        [--min_demos 20]          \\
        [--max_tasks 20]          \\
        [--max_episodes 50]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors import UniformExtractor, RandomExtractor

# (display_name, extractor_instance)
EXTRACTORS = [
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

try:
    from src.extractors.frame_diff import FrameDiffExtractor
    EXTRACTORS.append(("frame_diff", FrameDiffExtractor()))
except ImportError:
    pass


def check_task(loader, task: str, max_episodes: int) -> dict:
    """Run all extractors on up to max_episodes episodes for a given task."""
    ep_indices = loader.list_episodes(task)[:max_episodes]
    n_eps = len(ep_indices)

    per_extractor: dict = {name: {"kf_counts": [], "crs": []} for name, _ in EXTRACTORS}

    for ep_idx in ep_indices:
        episode = loader.load_episode(ep_idx)
        images = episode["images"]  # (T, H, W, 3)
        T = images.shape[0]

        for name, ext in EXTRACTORS:
            kf = ext.extract(images)
            per_extractor[name]["kf_counts"].append(len(kf))
            per_extractor[name]["crs"].append(len(kf) / T if T > 0 else 0.0)

    result = {"task": task, "n_episodes": n_eps, "extractors": {}}
    for name, data in per_extractor.items():
        kf_counts = data["kf_counts"]
        crs       = data["crs"]
        mean_kf   = float(np.mean(kf_counts))
        std_kf    = float(np.std(kf_counts))
        result["extractors"][name] = {
            "mean_kf": mean_kf,
            "std_kf":  std_kf,
            "min_kf":  int(np.min(kf_counts)),
            "max_kf":  int(np.max(kf_counts)),
            "cv_kf":   std_kf / mean_kf if mean_kf > 0 else 0.0,
            "mean_cr": float(np.mean(crs)),
            "std_cr":  float(np.std(crs)),
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         default="~/.cache/lerobot",
                        help="Local lerobot dataset cache root")
    parser.add_argument("--min_demos",    type=int, default=20,
                        help="Minimum episodes per task to include")
    parser.add_argument("--max_tasks",    type=int, default=20,
                        help="Maximum number of tasks to evaluate")
    parser.add_argument("--max_episodes", type=int, default=50,
                        help="Maximum episodes per task to use")
    args = parser.parse_args()

    from src.data.bridge_loader import BridgeDataLoader

    print(f"Loading BridgeDataLoader from {args.root} ...")
    loader = BridgeDataLoader(root=args.root)

    tasks = loader.list_tasks(min_demos=args.min_demos)[: args.max_tasks]
    if not tasks:
        sys.exit(f"No tasks with >= {args.min_demos} episodes found.")

    ext_names = [name for name, _ in EXTRACTORS]
    print(f"\nFound {len(tasks)} tasks  ·  extractors: {ext_names}\n")

    all_results: dict = {}
    for i, task in enumerate(tasks, 1):
        short = task[:55]
        sys.stdout.write(f"  [{i:>2}/{len(tasks)}] {short:<55}... ")
        sys.stdout.flush()
        result = check_task(loader, task, args.max_episodes)
        all_results[task] = result
        print(f"done ({result['n_episodes']} eps)")

    # --- summary table ---
    HDR = (
        f"{'TASK':<40} {'EXTRACTOR':<18} "
        f"{'N':>4} {'MEAN_KF':>8} {'STD_KF':>7} {'CV':>6} {'MEAN_CR':>8}"
    )
    SEP = "-" * len(HDR)
    print(f"\n{SEP}\n{HDR}\n{SEP}")
    for task, task_data in all_results.items():
        short = task[:39]
        for ext_name, s in task_data["extractors"].items():
            print(
                f"{short:<40} {ext_name:<18} {task_data['n_episodes']:>4} "
                f"{s['mean_kf']:>8.1f} {s['std_kf']:>7.2f} "
                f"{s['cv_kf']:>6.3f} {s['mean_cr']:>8.3f}"
            )
        print()

    # --- aggregate across all tasks ---
    print(SEP)
    print("AGGREGATE (mean across all tasks)")
    print(SEP)
    for ext_name in ext_names:
        all_mean_kf = [r["extractors"][ext_name]["mean_kf"] for r in all_results.values()]
        all_cv      = [r["extractors"][ext_name]["cv_kf"]   for r in all_results.values()]
        all_cr      = [r["extractors"][ext_name]["mean_cr"] for r in all_results.values()]
        print(
            f"{'(all tasks)':<40} {ext_name:<18} {'':>4} "
            f"{np.mean(all_mean_kf):>8.1f} {'':>7} {np.mean(all_cv):>6.3f} "
            f"{np.mean(all_cr):>8.3f}"
        )

    # --- save JSON ---
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "consistency_check_bridge.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
