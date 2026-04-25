# Keyframe Selection for Robot Learning

## Research Context

Training VLA models on robot demonstrations is expensive because most frames are redundant mid-motion states, yet no existing paper systematically compares uniform, heuristic, and learned keyframe selection at multiple compression ratios on a common manipulation benchmark. The closest prior work either predates modern VLA architectures (Wen et al., NeurIPS 2021) or performs a single uniform-vs-keyframe comparison in a narrow domain (Teleoperation KF IL, ASCE 2024). This project fills that gap on LIBERO using OpenVLA-7B as the downstream model.

## Problem Statement

Training VLA models (like OpenVLA) on robot demos is expensive because demos are long (~150 frames) but redundant — most frames show the arm mid-motion with nothing interesting happening. **Keyframe selection** compresses each demo to only the important moments (e.g. "just before grasp", "just after placing"), reducing training data size by 87–97% while (hopefully) preserving task-critical information.

The research question: **which keyframe selection method produces the best fine-tuned policy?**

## Method Taxonomy

From the literature survey (full details in `RESEARCH_final.md`):

```
Keyframe selection methods
├── Signal-based heuristics
│   ├── Velocity near-zero (joint/EE/hand): NoTVLA, RoboPrompt, SeeDo, VLA-RL
│   ├── Gripper state change: VLA-Thinker, RoboPrompt
│   └── Trajectory geometry (AWE): linear approximation error
├── Spatial / Learned
│   ├── 3D point cloud salient points: SPHINX
│   ├── 2D tracked points from internet video: ATM
│   └── Kinematics + diffusion predictor: Keyframe-Guided RL
├── Semantic / LLM-driven
│   ├── VLM subtask decomposition: RoboEnvision, VLA-Thinker
│   └── LLM frame scoring: KeyVideoLLM
└── Manual / Specified
    ├── User pose targets: RobotKeyframing, Semantic LfD
    └── Kinesthetic teaching: Akgun 2012
```

This project implements and compares the **signal-based heuristic** tier, which is the most portable and architecture-agnostic layer of the taxonomy.

## The 4 Extractors

All are implemented in `src/extractors/` and share the `KeyframeExtractor` interface: accept a trajectory array, return sorted integer indices always including frame 0 and T-1.

| Name | Logic | Input field | Key citations |
|---|---|---|---|
| `UniformExtractor` | N evenly spaced frames | any | AWE (CoRL 2023), Wen 2021 |
| `VelocityZeroExtractor` | Frames where EE speed < adaptive p25 threshold | `ee_vel` | NoTVLA (2025), RoboPrompt (2024), VLA-RL (2025), SeeDo (2024) |
| `GripperStateExtractor` | Frames at gripper open↔close transitions | `gripper_state` | VLA-Thinker (2026), RoboPrompt (2024) |
| `AWEExtractor` | Minimal waypoints for ε-accurate linear reconstruction of EE path | `ee_pos` | Zhao et al., CoRL 2023 (arXiv:2307.14326 ✅) |

## Data

8 HDF5 files from the LIBERO `libero_spatial` suite, each containing 50 expert demonstrations of a pick-and-place task ("pick up the black bowl from X and place it on the plate"). All tasks share the same object and robot; the spatial relationship between the bowl and reference objects varies.

Each demo stores:
- `ee_pos` (T×3) — end-effector XYZ in metres
- `ee_vel` (T×3) — computed via finite differences (not stored natively in LIBERO)
- `gripper_state` (T,) — abs mean finger qpos; ~0.04 = open, ~0.007 = closed
- `actions` (T×7) — delta actions `[dx, dy, dz, droll, dpitch, dyaw, gripper∈{-1,+1}]`
- `images` (T×128×128×3) — agentview RGB frames

Total: 400 demos × ~115–150 frames each.

## Consistency Results

Ran all 4 extractors on all 400 demos (8 tasks × 50 demos). Results are stable across tasks.

| Extractor | avg keyframes | CV (std/mean) ↓ | avg compression ratio |
|---|---|---|---|
| `uniform_10` | 10.0 | 0.000 | 8.6% |
| `velocity_p25_d5` | 8.5 | 0.150 | 7.1% |
| `gripper_d5` | **3.9** | 0.170 | 3.3% |
| `awe_eps0.01` | 14.3 | 0.124 | 12.0% |

Notable: gripper is the most aggressive extractor (~4 keyframes/demo) but also the most variable — it captures transitions only, missing the approach phase entirely on demos where the bowl is already grasped in an atypical way. Velocity and AWE are better-balanced for BC training.

Full per-task breakdown saved to `results/consistency_check.json`.

## Training Harness

OpenVLA normally trains on RLDS/TFDS format. We bypass that with a direct HDF5 reader.

**`openvla/prismatic/vla/datasets/libero_dataset.py`** — `LiberoKeyframeDataset`
- Map-style `torch.utils.data.Dataset`; pre-loads all keyframe `(image, action, instruction)` tuples at init (~50 MB/task, safe for LIBERO scale)
- Computes per-dimension q01/q99 from the training split; normalizes actions to [-1, 1]
- Exposes `dataset_statistics` in the format expected by `save_dataset_statistics()` for checkpoint de-normalization at inference
- Extractor is injected at construction; any `KeyframeExtractor` subclass works

**`openvla/vla-scripts/finetune_libero.py`** — training script
- Drop-in replacement for `finetune.py`; same LoRA loop, same W&B logging
- Extractor configured via CLI: `--extractor_name uniform | velocity_zero | gripper_state | awe`
- `--task_filter` is a comma-separated list of substrings to select a subset of tasks from `--hdf5_dir`

**`keyframe-selector/scripts/run_experiment.py`** — experiment launcher
- Takes two task name substrings, prints per-task consistency stats across all 4 extractors, emits ready-to-run `torchrun` commands for all three main conditions

## Experiment Design

### Conditions

| Condition | Extractor | Rationale |
|---|---|---|
| A — Uniform | `UniformExtractor(n=10)` | Standard baseline used in AWE, Wen 2021 |
| B — Velocity | `VelocityZeroExtractor(p=25)` | Dominant heuristic in NoTVLA, RoboPrompt, VLA-RL |
| C — AWE | `AWEExtractor(eps=0.01)` | Geometry-aware; most studied plug-in method |
| D — Gripper | `GripperStateExtractor()` | Supplementary; may underfit due to low keyframe count |

### Compression ratio sweep

Use AWE's `error_threshold` to vary CR continuously — lower `eps` = more waypoints = higher CR. Suggested sweep: `eps ∈ {0.005, 0.01, 0.02, 0.05}` covering CR ≈ 6–25%. Plot success rate vs. CR for each extractor to produce a compression–performance curve.

### Evaluation

- **Primary metric:** task success rate in LIBERO simulation (binary, averaged over 50 rollouts per checkpoint)
- **Always include a random-frame baseline** at each compression ratio, per the "Performance over Random" protocol (ACM Multimedia 2020). Report results as improvement over random, not absolute numbers only — this controls for the trivially-recoverable information present in any random frame subset.
- **Secondary metrics:** training loss convergence speed (steps to 50% success), action L1 loss at equivalent compression ratios

### Tasks for first experiment

Two `libero_spatial` tasks with distinct spatial configurations:
- `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`
- `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`

Run `python scripts/run_experiment.py` to get the exact `torchrun` commands.

## What's Next

- **Training** — requires a GPU (finetune_libero.py asserts CUDA). Copy the `torchrun` commands from `run_experiment.py` to a GPU machine and run conditions A, B, C.
- **Random baseline** — add `RandomExtractor(n)` to `src/extractors/` so every experiment run also trains a matched-N random-frame condition automatically.
- **Evaluation loop** — write a rollout script that loads a checkpoint and runs 50 LIBERO episodes, returning success rate.
- **Gripper extractor** — 4 keyframes/demo is likely too few for BC; consider a fallback that adds velocity-based frames between gripper transitions to bring CR closer to uniform's 8–10%.
- **Compression sweep** — after the first 2-task experiment validates the pipeline, sweep AWE `eps` across all 8 tasks.

## Literature

See `RESEARCH_final.md` for the full survey (25 papers, verification status, per-paper project relevance notes, and the research gap argument).
