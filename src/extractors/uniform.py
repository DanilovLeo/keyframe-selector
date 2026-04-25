import numpy as np

from .base import KeyframeExtractor


class UniformExtractor(KeyframeExtractor):
    """Uniform subsampling baseline: selects n evenly spaced frames.

    Always includes the first and last frame. This is the standard
    random/uniform baseline used to benchmark all other methods against.
    """

    def __init__(self, n_keyframes: int) -> None:
        """
        Args:
            n_keyframes: Total number of keyframes to select (including
                         the mandatory first and last frames). Must be >= 2.
        """
        if n_keyframes < 2:
            raise ValueError("n_keyframes must be >= 2 to include both endpoints.")
        self._n = n_keyframes

    @property
    def name(self) -> str:
        return f"uniform_{self._n}"

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Select n evenly spaced indices from the trajectory.

        Args:
            trajectory: Array of shape (T, D). Only T (length) is used.

        Returns:
            Sorted integer array of n indices spanning [0, T-1].
        """
        T = len(trajectory)
        if T <= self._n:
            return np.arange(T, dtype=int)
        return np.linspace(0, T - 1, self._n, dtype=int)
