"""
evaluate.py

Runs rollout evaluation of a fine-tuned OpenVLA checkpoint on a LIBERO task.

For each rollout:
  1. Reset the LIBERO sim to the initial state from the HDF5 demo file.
  2. Run the policy until the task succeeds (done=True) or max_steps is reached.
  3. Aggregate success rate over n_rollouts.

Usage:
    python scripts/evaluate.py \\
        --run_dir ~/keyframe_selection/runs/<checkpoint_dir> \\
        --hdf5_path ~/keyframe_selection/LIBERO/libero/datasets/libero_spatial/<task>_demo.hdf5 \\
        --libero_dir ~/keyframe_selection/LIBERO/libero \\
        [--n_rollouts 50] \\
        [--max_steps 600] \\
        [--output results.json]

--run_dir  must contain:
    - dataset_statistics.json (written by save_dataset_statistics during training)
    - config.json + model weights (the merged LoRA checkpoint)

--hdf5_path provides the BDDL file path (from attrs) and the per-demo initial
    sim states used to seed the rollouts.  Must be a LIBERO _demo.hdf5 file
    from the same task suite used during training.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# --- OpenVLA custom class registration ---
# Must happen before any AutoXxx.from_pretrained call on an OpenVLA checkpoint.
_openvla_dir = Path(__file__).parent.parent.parent / "openvla"
if _openvla_dir.exists() and str(_openvla_dir) not in sys.path:
    sys.path.insert(0, str(_openvla_dir))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, model_path: str) -> str:
    if "v01" in model_path:
        return (
            f"{SYSTEM_PROMPT} USER: What action should the robot take to "
            f"{instruction.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def load_task_info(hdf5_path: str):
    """Extract language instruction, BDDL file name, and per-demo init states."""
    with h5py.File(os.path.expanduser(hdf5_path), "r") as f:
        attrs = dict(f["data"].attrs)
        problem_info = json.loads(attrs["problem_info"])
        instruction = problem_info.get("language_instruction", Path(hdf5_path).stem)

        # BDDL file stored as a relative path from the LIBERO repo root, e.g.
        # "libero/libero/bddl_files/libero_spatial/pick_up_...bddl"
        bddl_rel = attrs.get("bddl_file_name", "")

        demo_keys = sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
        # states[0] is the full MuJoCo sim state at the start of each demo
        init_states = [f[f"data/{k}/states"][0] for k in demo_keys]

    return instruction, bddl_rel, init_states


def resolve_bddl(bddl_rel: str, libero_dir: str) -> str:
    """Return an absolute path to the BDDL file.

    bddl_rel is stored as a path relative to the LIBERO repo root (the parent
    of the `libero` package directory).  We try two common layouts:
      1. libero_dir / bddl_rel
      2. libero_dir / basename(bddl_rel)  (glob fallback)
    """
    libero_dir = Path(libero_dir).expanduser()

    # Try the relative path stored in the HDF5 attrs
    candidate = libero_dir / bddl_rel
    if candidate.exists():
        return str(candidate)

    # Fallback: the stored path may use an old repo layout; just match by filename
    bddl_name = Path(bddl_rel).name
    matches = list(libero_dir.rglob(bddl_name))
    if matches:
        return str(matches[0])

    raise FileNotFoundError(
        f"Cannot find BDDL file '{bddl_rel}' under {libero_dir}.\n"
        "Pass --libero_dir pointing at the root of the LIBERO repo."
    )


def load_model(run_dir: str, device: torch.device):
    """Load the merged OpenVLA checkpoint and its dataset statistics."""
    run_dir = Path(run_dir).expanduser()
    processor = AutoProcessor.from_pretrained(str(run_dir), trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        str(run_dir),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()

    stats_path = run_dir / "dataset_statistics.json"
    with open(stats_path) as f:
        vla.norm_stats = json.load(f)

    unnorm_key = next(iter(vla.norm_stats))  # single dataset per run
    return vla, processor, unnorm_key


@torch.inference_mode()
def predict_action(
    vla,
    processor,
    image: np.ndarray,
    instruction: str,
    unnorm_key: str,
    model_path: str,
    device: torch.device,
) -> np.ndarray:
    prompt = get_openvla_prompt(instruction, model_path)
    inputs = processor(prompt, Image.fromarray(image).convert("RGB")).to(
        device, dtype=torch.bfloat16
    )
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action  # np.ndarray (7,)


def run_rollouts(
    vla,
    processor,
    unnorm_key: str,
    model_path: str,
    device: torch.device,
    bddl_file: str,
    instruction: str,
    init_states: list,
    n_rollouts: int,
    max_steps: int,
) -> dict:
    # Import here so the script is importable on machines without LIBERO installed
    # (e.g., for unit-testing the rest of the module).
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError(
            "LIBERO is not on sys.path. Add the LIBERO repo to PYTHONPATH or "
            "pass --libero_dir so the script can locate it."
        )

    env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)

    n_rollouts = min(n_rollouts, len(init_states))
    successes = 0

    for i in tqdm(range(n_rollouts), desc="rollouts"):
        env.reset()
        obs = env.set_init_state(init_states[i])

        # Settle the sim for a few frames before starting the policy
        for _ in range(5):
            obs, _, _, _ = env.step([0.0] * 7)

        done = False
        for _ in range(max_steps):
            image = obs["agentview_image"]  # (128, 128, 3) uint8
            action = predict_action(
                vla, processor, image, instruction, unnorm_key, model_path, device
            )
            obs, _reward, done, _info = env.step(action.tolist())
            if done:
                break

        if done:
            successes += 1

    env.close()
    return {"n_rollouts": n_rollouts, "successes": successes, "success_rate": successes / n_rollouts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir",    required=True,
                        help="Path to the merged OpenVLA checkpoint directory")
    parser.add_argument("--hdf5_path",  required=True,
                        help="Path to the LIBERO _demo.hdf5 file for the task to evaluate")
    parser.add_argument("--libero_dir", default="~/keyframe_selection/LIBERO/libero",
                        help="Path to the LIBERO repo root (parent of the libero package)")
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--max_steps",  type=int, default=600)
    parser.add_argument("--output",     default="",
                        help="Path for JSON output (default: run_dir/eval_results.json)")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "Evaluation requires a GPU."
    device = torch.device("cuda:0")

    run_dir = Path(args.run_dir).expanduser()
    libero_dir = Path(args.libero_dir).expanduser()

    # Add LIBERO to sys.path so OffScreenRenderEnv is importable
    if str(libero_dir) not in sys.path:
        sys.path.insert(0, str(libero_dir))

    print(f"Loading model from {run_dir} ...")
    vla, processor, unnorm_key = load_model(str(run_dir), device)
    print(f"  unnorm_key: {unnorm_key}")

    instruction, bddl_rel, init_states = load_task_info(args.hdf5_path)
    bddl_file = resolve_bddl(bddl_rel, str(libero_dir))
    print(f"Task: {instruction}")
    print(f"BDDL: {bddl_file}")
    print(f"Init states: {len(init_states)} demos available, using {args.n_rollouts}")

    results = run_rollouts(
        vla=vla,
        processor=processor,
        unnorm_key=unnorm_key,
        model_path=str(run_dir),
        device=device,
        bddl_file=bddl_file,
        instruction=instruction,
        init_states=init_states,
        n_rollouts=args.n_rollouts,
        max_steps=args.max_steps,
    )

    results["run_dir"] = str(run_dir)
    results["task"] = instruction
    results["unnorm_key"] = unnorm_key

    print(
        f"\nSuccess rate: {results['successes']}/{results['n_rollouts']} "
        f"= {results['success_rate']:.1%}"
    )

    out_path = Path(args.output) if args.output else run_dir / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
