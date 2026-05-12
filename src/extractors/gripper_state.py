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


class GripperFallbackExtractor(KeyframeExtractor):
    """Gripper transitions padded with velocity near-zero frames.

    Finds gripper open/close transitions first. If the total count is below
    min_n, fills the remaining slots with the slowest (velocity near-zero)
    frames from the gaps between transitions, bringing coverage closer to
    the uniform/velocity extractors (~8-10 frames/demo).

    Accepts the combined "gripper_vel" trajectory from load_libero_demo():
        col 0   — gripper scalar (abs-mean finger qpos)
        cols 1:4 — EE velocity (m/frame, finite-difference)

    Based on the observation in the OVERVIEW that plain GripperStateExtractor
    (~4 frames/demo) misses the approach phase entirely.
    """

    def __init__(self, min_n: int = 8, gripper_min_dist: int = 5) -> None:
        """
        Args:
            min_n:            Target minimum number of keyframes. Gripper
                              transitions are always included; velocity frames
                              are added until this count is reached.
            gripper_min_dist: Minimum frames between consecutive gripper
                              transition keyframes (passed to inner extractor).
        """
        self._min_n = min_n
        self._gripper_min_dist = gripper_min_dist
        self._gripper = GripperStateExtractor(min_dist=gripper_min_dist)

    @property
    def name(self) -> str:
        return f"gripper_fallback_n{self._min_n}"

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Select gripper-transition keyframes, padded with velocity frames.

        Args:
            trajectory: (T, 4) array — col 0 is gripper_state, cols 1:4 are
                        ee_vel. Use traj_key="gripper_vel" with this extractor.

        Returns:
            Sorted integer array of keyframe indices including endpoints.
        """
        gripper = trajectory[:, 0]
        ee_vel = trajectory[:, 1:4]
        T = len(gripper)

        kf_set = set(self._gripper.extract(gripper).tolist())

        if len(kf_set) >= self._min_n:
            return np.array(sorted(kf_set), dtype=int)

        needed = self._min_n - len(kf_set)
        speed = np.linalg.norm(ee_vel, axis=1)

        # Fill from non-selected interior frames, slowest speed first
        candidates = sorted(
            (i for i in range(1, T - 1) if i not in kf_set),
            key=lambda i: speed[i],
        )
        for i in candidates[:needed]:
            kf_set.add(i)

        return np.array(sorted(kf_set), dtype=int)
