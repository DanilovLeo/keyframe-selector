import numpy as np

from .base import KeyframeExtractor


class GripperStateExtractor(KeyframeExtractor):
    """Selects frames at gripper open/close transitions.

    Gripper qpos is reduced to a scalar (mean across fingers), binarized
    around its mid-range, and state changes are detected as keyframes.
    A min_dist guard prevents duplicate keyframes near the same transition.

    Based on:
        VLA-Thinker (arXiv:2603.14523), RoboPrompt (arXiv:2410.12782).
    """

    def __init__(self, min_dist: int = 5) -> None:
        """
        Args:
            min_dist: Minimum number of frames between consecutive keyframes.
        """
        self._min_dist = min_dist

    @property
    def name(self) -> str:
        return f"gripper_state_d{self._min_dist}"

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Select frames at gripper open/close transitions.

        Args:
            trajectory: 1-D scalar gripper signal of shape (T,), as returned
                        by load_libero_demo()["gripper_state"].
                        Values are abs-mean finger qpos: ~0.040 = open,
                        ~0.007 = closed. Binarized at the midpoint of the
                        observed range (not a fixed 0.5, because the real
                        signal lives in [0.007, 0.040], not [0, 1]).

        Returns:
            Sorted integer array of keyframe indices including endpoints.
        """
        T = len(trajectory)
        if T < 2:
            return np.array([0], dtype=int)

        scalar = np.asarray(trajectory, dtype=float)

        lo, hi = scalar.min(), scalar.max()
        if hi - lo < 1e-6:
            # Gripper never moves; return only endpoints
            return np.array([0, T - 1], dtype=int)

        # Dynamic midpoint binarization (works with any unit or scale)
        mid = (lo + hi) / 2.0
        binary = (scalar > mid).astype(int)

        selected = [0]
        for i in range(1, T):
            if binary[i] != binary[i - 1] and (i - selected[-1]) >= self._min_dist:
                selected.append(i)

        if selected[-1] != T - 1:
            selected.append(T - 1)

        return np.array(selected, dtype=int)
