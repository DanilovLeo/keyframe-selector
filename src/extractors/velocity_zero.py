from __future__ import annotations

import numpy as np

from .base import KeyframeExtractor


class VelocityZeroExtractor(KeyframeExtractor):
    """Selects frames where end-effector speed drops below a threshold.

    Supports two threshold modes:

    **Fixed mode** (``threshold`` is a float):
        Frames whose speed is below the given value are candidates.
        Use this when you have task-specific knowledge of the speed scale
        (e.g. threshold=0.005 for slow LIBERO pick-and-place demos).

    **Adaptive mode** (``threshold=None``, default):
        The threshold is set to ``np.percentile(speed, percentile)`` on
        each call to ``extract()``, adapting to the actual speed distribution
        of the demo.  The default ``percentile=25`` targets the bottom
        quartile of frames, i.e. the quarter of frames where the EE is
        moving most slowly — a data-driven proxy for "paused" moments.
        This is more robust than a fixed value when the absolute speed
        range varies across tasks or control frequencies.

    After ``extract()`` returns, the attribute ``threshold_used`` holds
    the threshold that was actually applied, regardless of mode.

    Based on:
        NoTVLA (arXiv:2510.03895), RoboPrompt (arXiv:2410.12782),
        VLA-RL (arXiv:2505.18719).
    """

    def __init__(
        self,
        threshold: float | None = None,
        percentile: float = 25.0,
        min_dist: int = 5,
    ) -> None:
        """
        Args:
            threshold:  Fixed speed threshold (m/frame). When ``None``,
                        the threshold is derived from the data each call.
            percentile: Percentile of the speed distribution used as the
                        threshold in adaptive mode. Ignored when ``threshold``
                        is not ``None``. percentile=25 means "slower than
                        75% of all frames in this demo".
            min_dist:   Minimum number of frames between consecutive keyframes.
        """
        self._threshold = threshold
        self._percentile = percentile
        self._min_dist = min_dist
        self.threshold_used: float = float("nan")

    @property
    def name(self) -> str:
        if self._threshold is None:
            return f"velocity_zero_p{self._percentile}_d{self._min_dist}"
        return f"velocity_zero_t{self._threshold}_d{self._min_dist}"

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Select frames where EE speed is near zero.

        Sets ``self.threshold_used`` to the threshold applied this call.

        Args:
            trajectory: EE velocity array of shape (T, 3), as returned by
                        ``load_libero_demo()["ee_vel"]``. Speed is the L2
                        norm across the 3 components.

        Returns:
            Sorted integer array of keyframe indices including both endpoints.
        """
        T = len(trajectory)
        if T < 2:
            self.threshold_used = self._threshold if self._threshold is not None else 0.0
            return np.array([0], dtype=int)

        speed = np.linalg.norm(trajectory, axis=1)   # (T,)

        if self._threshold is None:
            thresh = float(np.percentile(speed, self._percentile))
        else:
            thresh = float(self._threshold)

        self.threshold_used = thresh

        selected = [0]
        for i in range(1, T - 1):
            if speed[i] < thresh and (i - selected[-1]) >= self._min_dist:
                selected.append(i)

        if selected[-1] != T - 1:
            selected.append(T - 1)

        return np.array(selected, dtype=int)
