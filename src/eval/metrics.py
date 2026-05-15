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


