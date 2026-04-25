"""Evaluation metrics for keyframe selection experiments."""

import numpy as np


def compression_ratio(n_original: int, n_selected: int) -> float:
    """Fraction of frames retained after keyframe selection.

    Lower values mean higher compression (fewer frames kept).

    Args:
        n_original: Total number of frames in the original trajectory.
        n_selected: Number of selected keyframes.

    Returns:
        n_selected / n_original, or 0.0 if n_original is 0.
    """
    if n_original == 0:
        return 0.0
    return n_selected / n_original


def coverage_score(keyframe_indices: np.ndarray, n_total: int) -> float:
    """Temporal coverage: fraction of the trajectory spanned by keyframes.

    Defined as (last_keyframe - first_keyframe) / (n_total - 1).
    A value of 1.0 means keyframes span the full trajectory; lower values
    indicate keyframes are clustered away from the endpoints.

    Args:
        keyframe_indices: 1-D integer array of selected frame indices.
        n_total:          Total number of frames in the trajectory.

    Returns:
        Coverage ratio in [0, 1].
    """
    if n_total <= 1 or len(keyframe_indices) == 0:
        return 0.0
    span = int(keyframe_indices.max()) - int(keyframe_indices.min())
    return span / (n_total - 1)


def task_phase_coverage(
    keyframe_indices: np.ndarray,
    grasp_idx: int,
    n_total: int,
) -> dict:
    """Break down keyframe coverage by task phase (pre-grasp vs. post-grasp).

    Args:
        keyframe_indices: 1-D integer array of selected frame indices.
        grasp_idx:        Frame index at which the gripper closes (grasp event).
                          Typically detected from gripper state transitions.
        n_total:          Total number of frames in the trajectory.

    Returns:
        Dictionary with keys:
            "pre_grasp_frac"  — fraction of keyframes before grasp_idx
            "post_grasp_frac" — fraction of keyframes from grasp_idx onward
            "pre_grasp_n"     — count of keyframes before grasp_idx
            "post_grasp_n"    — count of keyframes from grasp_idx onward
            "grasp_idx"       — the grasp_idx passed in
            "n_total"         — n_total passed in
    """
    if len(keyframe_indices) == 0:
        return {
            "pre_grasp_frac": 0.0,
            "post_grasp_frac": 0.0,
            "pre_grasp_n": 0,
            "post_grasp_n": 0,
            "grasp_idx": grasp_idx,
            "n_total": n_total,
        }

    n_kf = len(keyframe_indices)
    pre_n = int(np.sum(keyframe_indices < grasp_idx))
    post_n = n_kf - pre_n

    return {
        "pre_grasp_frac": pre_n / n_kf,
        "post_grasp_frac": post_n / n_kf,
        "pre_grasp_n": pre_n,
        "post_grasp_n": post_n,
        "grasp_idx": grasp_idx,
        "n_total": n_total,
    }
