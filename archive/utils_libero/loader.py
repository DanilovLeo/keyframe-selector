"""LIBERO HDF5 demo loader.

Real HDF5 layout (LIBERO libero-v1, verified against actual files):

    data/                            attrs: num_demos, problem_info (JSON)
      demo_0/
        actions          (T, 7)     [dx,dy,dz, droll,dpitch,dyaw, gripper∈{-1,+1}]
        rewards          (T,)       uint8
        dones            (T,)       uint8
        obs/
          ee_pos         (T, 3)     end-effector XYZ (metres)
          ee_states      (T, 6)     ee_pos (cols 0:3) + ee_ori axis-angle (cols 3:6)
          ee_ori         (T, 3)     axis-angle orientation (same as ee_states[:,3:])
          gripper_states (T, 2)     finger qpos: finger0 ∈ [+0.006, +0.040],
                                                  finger1 ∈ [-0.040, -0.006]
                                    — opposite signs; mean ≈ 0 always.
                                    Use abs().mean(axis=1) to get [0.007, 0.040]:
                                    ~0.040 = open, ~0.007 = closed.
          joint_states   (T, 7)     joint angles (rad)
          agentview_rgb  (T,128,128,3) uint8
          eye_in_hand_rgb(T,128,128,3) uint8
      demo_1/
        ...

NOTE: ee_states[:,3:] is orientation (axis-angle), NOT velocity.
      EE velocity is not stored; it is computed here via np.gradient(ee_pos).
"""

import json
import os
from pathlib import Path

import h5py
import numpy as np


def load_libero_demo(path: str, demo_idx: int = 0) -> dict:
    """Load a single LIBERO demonstration from an HDF5 file.

    EE velocity is not stored in LIBERO files; it is estimated via
    np.gradient (central differences, forward/backward at the endpoints).

    The gripper scalar is np.abs(gripper_states).mean(axis=1) because the
    two fingers have opposite signs (+/−), making the raw mean always ≈ 0.
    The resulting scalar ranges from ~0.007 (closed) to ~0.040 (open).

    Args:
        path:     Path to a LIBERO .hdf5 demo file.
        demo_idx: Zero-based index of the demonstration to load.

    Returns:
        Dictionary with keys:
            "ee_pos"        — (T, 3) end-effector XYZ position
            "ee_vel"        — (T, 3) finite-diff EE velocity (m/frame)
            "gripper_state" — (T,)   abs-mean of finger qpos; ~0.04=open, ~0.007=closed
            "joint_states"  — (T, 7) joint angles in radians
            "actions"       — (T, 7) delta actions; last column is gripper ∈ {-1, +1}
            "rewards"       — (T,)   per-step rewards
            "images"        — (T, 128, 128, 3) uint8 agentview RGB frames
            "n_demos"       — int total demonstrations in this file
            "task_name"     — str language instruction from problem_info attrs

    Raises:
        IndexError: If demo_idx >= n_demos.
    """
    with h5py.File(os.path.expanduser(path), "r") as f:
        demos = sorted(
            f["data"].keys(),
            key=lambda x: int(x.split("_")[1]),
        )
        n_demos = len(demos)
        if demo_idx >= n_demos:
            raise IndexError(
                f"demo_idx={demo_idx} out of range; file has {n_demos} demos."
            )

        problem_info = json.loads(f["data"].attrs["problem_info"])
        task_name: str = problem_info.get("language_instruction", Path(path).stem)

        ep = demos[demo_idx]
        grp = f[f"data/{ep}"]

        ee_pos: np.ndarray          = grp["obs/ee_pos"][()]
        gripper_states: np.ndarray  = grp["obs/gripper_states"][()]
        joint_states: np.ndarray    = grp["obs/joint_states"][()]
        actions: np.ndarray         = grp["actions"][()]
        rewards: np.ndarray         = grp["rewards"][()]
        images: np.ndarray          = grp["obs/agentview_rgb"][()]

    # Velocity: central differences (ee_states[:,3:] is orientation, not vel)
    ee_vel: np.ndarray = np.gradient(ee_pos, axis=0)

    # Gripper scalar: abs-mean removes the symmetric sign flip between fingers
    gripper_state: np.ndarray = np.abs(gripper_states).mean(axis=1)

    # Combined signal for GripperFallbackExtractor: col 0 = gripper_state, cols 1:4 = ee_vel
    gripper_vel: np.ndarray = np.column_stack([gripper_state, ee_vel])

    return {
        "ee_pos":        ee_pos,
        "ee_vel":        ee_vel,
        "gripper_state": gripper_state,
        "gripper_vel":   gripper_vel,
        "joint_states":  joint_states,
        "actions":       actions,
        "rewards":       rewards,
        "images":        images,
        "n_demos":       n_demos,
        "task_name":     task_name,
    }


def list_demos(path: str) -> list[str]:
    """Return sorted demo keys from an HDF5 file.

    Args:
        path: Path to a LIBERO .hdf5 demo file.

    Returns:
        List of demo group names, e.g. ["demo_0", "demo_1", ...].
    """
    with h5py.File(os.path.expanduser(path), "r") as f:
        return sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1]))
