"""
Multi-task consistency check: runs all 4 extractors on every demo across all 8 HDF5 files.

Metrics reported per (task, extractor):
  mean_kf / std_kf  — average and std of keyframe count across demos
  cv_kf             — coefficient of variation (std/mean); lower = more consistent
  mean_cr           — mean compression ratio (n_keyframes / T)

Output:
  stdout  — formatted table
  results/consistency_check.json — full results

Usage (from keyframe-selector/):
    python scripts/consistency_check.py [--dataset_dir PATH]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors import AWEExtractor, GripperStateExtractor, UniformExtractor, VelocityZeroExtractor
from src.utils.loader import list_demos, load_libero_demo

DEFAULT_DATASET_DIR = (
    "~/keyframe_selection/LIBERO/libero/datasets/libero_spatial"
)

# (display_name, extractor_instance, demo_field_to_pass)
EXTRACTORS = [
    ("uniform_10",       UniformExtractor(n_keyframes=10),                    "ee_pos"),
    ("velocity_p25_d5",  VelocityZeroExtractor(percentile=25, min_dist=5),    "ee_vel"),
    ("gripper_d5",       GripperStateExtractor(min_dist=5),                   "gripper_state"),
    ("awe_eps0.01",      AWEExtractor(error_threshold=0.01),                  "ee_pos"),
]


def check_file(hdf5_path: Path) -> dict:
    demo_keys = list_demos(str(hdf5_path))
    n_demos = len(demo_keys)
    task_result = {"path": str(hdf5_path), "n_demos": n_demos, "extractors": {}}

    for ext_name, extractor, traj_key in EXTRACTORS:
        kf_counts, crs = [], []
        for demo_idx in range(n_demos):
            demo = load_libero_demo(str(hdf5_path), demo_idx=demo_idx)
            traj = demo[traj_key]
            kf = extractor.extract(traj)
            T = len(traj)
            kf_counts.append(len(kf))
            crs.append(len(kf) / T if T > 0 else 0.0)

        mean_kf = float(np.mean(kf_counts))
        std_kf = float(np.std(kf_counts))
        task_result["extractors"][ext_name] = {
            "mean_kf": mean_kf,
            "std_kf":  std_kf,
            "min_kf":  int(np.min(kf_counts)),
            "max_kf":  int(np.max(kf_counts)),
            "cv_kf":   std_kf / mean_kf if mean_kf > 0 else 0.0,
            "mean_cr": float(np.mean(crs)),
            "std_cr":  float(np.std(crs)),
        }

    return task_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    hdf5_files = sorted(dataset_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"No HDF5 files found in {dataset_dir}")
        sys.exit(1)

    print(f"Found {len(hdf5_files)} HDF5 files in {dataset_dir}\n")

    all_results = {}
    for hdf5_path in hdf5_files:
        task_name = hdf5_path.stem.replace("_demo", "")
        sys.stdout.write(f"  Processing {task_name[:60]}... ")
        sys.stdout.flush()
        all_results[task_name] = check_file(hdf5_path)
        print(f"done ({all_results[task_name]['n_demos']} demos)")

    # --- summary table ---
    HDR = f"{'TASK':<42} {'EXTRACTOR':<20} {'N':>4} {'MEAN_KF':>8} {'STD_KF':>7} {'CV':>6} {'MEAN_CR':>8}"
    SEP = "-" * len(HDR)
    print(f"\n{SEP}\n{HDR}\n{SEP}")
    for task_name, task_data in all_results.items():
        short = task_name[:41]
        for ext_name, s in task_data["extractors"].items():
            print(
                f"{short:<42} {ext_name:<20} {task_data['n_demos']:>4} "
                f"{s['mean_kf']:>8.1f} {s['std_kf']:>7.2f} {s['cv_kf']:>6.3f} {s['mean_cr']:>8.3f}"
            )
        print()

    # --- per-extractor aggregate across all tasks ---
    print(f"{SEP}")
    print("AGGREGATE (mean across all tasks)")
    print(f"{SEP}")
    for ext_name, _, _ in EXTRACTORS:
        all_mean_kf = [r["extractors"][ext_name]["mean_kf"] for r in all_results.values()]
        all_cv      = [r["extractors"][ext_name]["cv_kf"]   for r in all_results.values()]
        all_cr      = [r["extractors"][ext_name]["mean_cr"] for r in all_results.values()]
        print(
            f"{'(all tasks)':<42} {ext_name:<20} {'':>4} "
            f"{np.mean(all_mean_kf):>8.1f} {'':>7} {np.mean(all_cv):>6.3f} {np.mean(all_cr):>8.3f}"
        )

    # --- save JSON ---
    out_path = Path(__file__).parent.parent / "results" / "consistency_check.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    main()
