"""
run_experiment.py

Prepares and validates the two-task uniform-vs-velocity training experiment.

Steps:
  1. Locate the two specified task HDF5 files.
  2. Run the consistency check for those tasks across all 4 extractors.
  3. Print the exact torchrun commands for both conditions.

Usage:
    python scripts/run_experiment.py \
        --task1 "black_bowl_from_table_center" \
        --task2 "black_bowl_next_to_plate" \
        [--dataset_dir PATH]   \
        [--openvla_dir PATH]   \
        [--run_root_dir PATH]  \
        [--n_gpus 1]

task1 / task2 are substrings matched against HDF5 filenames (case-sensitive).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors import AWEExtractor, GripperStateExtractor, UniformExtractor, VelocityZeroExtractor
from src.utils.loader import list_demos, load_libero_demo

DEFAULT_DATASET_DIR = (
    "~/keyframe_selection/LIBERO/libero/datasets/libero_spatial"
)
DEFAULT_OPENVLA_DIR = "~/keyframe_selection/openvla"
DEFAULT_RUN_ROOT = "~/keyframe_selection/runs"

EXTRACTORS = [
    ("uniform_10",       UniformExtractor(n_keyframes=10),                 "ee_pos"),
    ("velocity_p25_d5",  VelocityZeroExtractor(percentile=25, min_dist=5), "ee_vel"),
    ("gripper_d5",       GripperStateExtractor(min_dist=5),                "gripper_state"),
    ("awe_eps0.01",      AWEExtractor(error_threshold=0.01),               "ee_pos"),
]


def find_task_file(dataset_dir: Path, substring: str) -> Path:
    matches = [p for p in dataset_dir.glob("*.hdf5") if substring in p.stem]
    if not matches:
        raise FileNotFoundError(
            f"No HDF5 file containing '{substring}' found in {dataset_dir}.\n"
            f"Available: {sorted(p.stem for p in dataset_dir.glob('*.hdf5'))}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous match for '{substring}': {[p.stem for p in matches]}\n"
            "Provide a more specific substring."
        )
    return matches[0]


def consistency_for_file(hdf5_path: Path) -> dict:
    demo_keys = list_demos(str(hdf5_path))
    results = {}
    for ext_name, extractor, traj_key in EXTRACTORS:
        kf_counts = []
        for demo_idx in range(len(demo_keys)):
            demo = load_libero_demo(str(hdf5_path), demo_idx=demo_idx)
            kf = extractor.extract(demo[traj_key])
            kf_counts.append(len(kf))
        mean = float(np.mean(kf_counts))
        results[ext_name] = {
            "mean_kf": mean,
            "std_kf":  float(np.std(kf_counts)),
            "cv":      float(np.std(kf_counts) / mean) if mean > 0 else 0.0,
            "cr":      mean / len(demo["ee_pos"]),
        }
    return {"n_demos": len(demo_keys), "extractors": results}


def print_consistency(task_name: str, stats: dict) -> None:
    print(f"\n  Task: {task_name}  ({stats['n_demos']} demos)")
    print(f"  {'Extractor':<22} {'mean_kf':>8} {'std_kf':>8} {'cv':>6} {'cr':>7}")
    print("  " + "-" * 56)
    for ext_name, s in stats["extractors"].items():
        print(
            f"  {ext_name:<22} {s['mean_kf']:>8.1f} {s['std_kf']:>8.2f} {s['cv']:>6.3f} {s['cr']:>7.3f}"
        )


def torchrun_cmd(
    openvla_dir: Path,
    hdf5_dir: Path,
    task_filter: str,
    extractor_name: str,
    run_root: Path,
    n_gpus: int,
    extra_tag: str = "",
) -> str:
    script = openvla_dir / "vla-scripts" / "finetune_libero.py"
    tag = f"--run_id_note {extra_tag}" if extra_tag else ""
    return (
        f"torchrun --standalone --nnodes 1 --nproc-per-node {n_gpus} \\\n"
        f"    {script} \\\n"
        f"    --hdf5_dir {hdf5_dir} \\\n"
        f"    --task_filter \"{task_filter}\" \\\n"
        f"    --extractor_name {extractor_name} \\\n"
        f"    --run_root_dir {run_root} \\\n"
        f"    --batch_size 8 \\\n"
        f"    --max_steps 10000 \\\n"
        f"    --save_steps 2500 \\\n"
        f"    {tag}"
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1",        default="black_bowl_from_table_center")
    parser.add_argument("--task2",        default="black_bowl_next_to_the_plate_and_place")
    parser.add_argument("--dataset_dir",  default=DEFAULT_DATASET_DIR)
    parser.add_argument("--openvla_dir",  default=DEFAULT_OPENVLA_DIR)
    parser.add_argument("--run_root_dir", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--n_gpus",       type=int, default=1)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).expanduser()
    openvla_dir = Path(args.openvla_dir).expanduser()
    run_root    = Path(args.run_root_dir).expanduser()

    print("=" * 70)
    print("  LIBERO keyframe-selection experiment setup")
    print("=" * 70)

    # Locate HDF5 files
    file1 = find_task_file(dataset_dir, args.task1)
    file2 = find_task_file(dataset_dir, args.task2)
    print(f"\nTask 1: {file1.stem}")
    print(f"Task 2: {file2.stem}")

    # Consistency check
    print("\n--- Keyframe consistency check ---")
    stats1 = consistency_for_file(file1)
    stats2 = consistency_for_file(file2)
    print_consistency(file1.stem, stats1)
    print_consistency(file2.stem, stats2)

    # Combined filter string for training both tasks together
    combined_filter = f"{args.task1},{args.task2}"

    # Print training commands
    print("\n" + "=" * 70)
    print("  TRAINING COMMANDS")
    print("=" * 70)
    print("\n[Condition A]  uniform_10 keyframes")
    print("-" * 70)
    print(torchrun_cmd(openvla_dir, dataset_dir, combined_filter, "uniform",
                       run_root, args.n_gpus, extra_tag="uniform10_2task"))

    print("\n[Condition B]  velocity_zero p25 keyframes")
    print("-" * 70)
    print(torchrun_cmd(openvla_dir, dataset_dir, combined_filter, "velocity_zero",
                       run_root, args.n_gpus, extra_tag="velp25_2task"))

    print("\n[Optional C]  awe_eps0.01 keyframes")
    print("-" * 70)
    print(torchrun_cmd(openvla_dir, dataset_dir, combined_filter, "awe",
                       run_root, args.n_gpus, extra_tag="awe001_2task"))

    print("\nRun each command on a GPU machine from the openvla/ directory.")
    print("Checkpoints and dataset_statistics.json will be saved under --run_root_dir.")


if __name__ == "__main__":
    main()
