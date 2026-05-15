import numpy as np

from .base import KeyframeExtractor


class AWEExtractor(KeyframeExtractor):
    """Automatic Waypoint Extraction via linear trajectory approximation.

    Greedily finds the minimal set of waypoints such that the piecewise-linear
    reconstruction of the full trajectory stays within error_threshold at every
    intermediate frame. Extends each segment as far as possible before the
    reconstruction error exceeds the threshold, then starts a new segment.

    Based on:
        AWE — Automatic Waypoint Extraction (arXiv:2307.14326, CoRL 2023).
    """

    def __init__(self, error_threshold: float = 0.01) -> None:
        """
        Args:
            error_threshold: Maximum allowed per-frame L2 reconstruction error.
                             In metres when trajectory is EE XYZ position.
        """
        self._eps = error_threshold

    @property
    def name(self) -> str:
        return f"awe_eps{self._eps}"

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Select minimal waypoints for ε-accurate linear reconstruction.

        Args:
            trajectory: Array of shape (T, D), typically EE XYZ position.

        Returns:
            Sorted integer array of waypoint indices including both endpoints.
        """
        T = len(trajectory)
        if T < 2:
            return np.arange(T, dtype=int)

        selected = [0]
        seg_start = 0

        while seg_start < T - 1:
            last_valid = seg_start + 1

            for candidate_end in range(seg_start + 2, T):
                # Points strictly between seg_start and candidate_end
                n_inner = candidate_end - seg_start - 1
                t = np.arange(1, n_inner + 1) / (candidate_end - seg_start)  # (n_inner,)
                interp = (
                    (1.0 - t[:, None]) * trajectory[seg_start]
                    + t[:, None] * trajectory[candidate_end]
                )
                actual = trajectory[seg_start + 1 : candidate_end]
                max_err = np.max(np.linalg.norm(actual - interp, axis=-1))
                if max_err > self._eps:
                    break
                last_valid = candidate_end
            else:
                # Reached T-1 without exceeding threshold
                last_valid = T - 1

            selected.append(last_valid)
            seg_start = last_valid

        if selected[-1] != T - 1:
            selected.append(T - 1)

        return np.array(sorted(set(selected)), dtype=int)
