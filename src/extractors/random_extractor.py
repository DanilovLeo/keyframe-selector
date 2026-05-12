import numpy as np

from .base import KeyframeExtractor


class RandomExtractor(KeyframeExtractor):
    """Random frame selection baseline for matched-N comparisons.

    Always includes the first and last frame. For any compression ratio
    experiment, pair this with the heuristic extractor at the same N to
    verify the method beats chance selection.

    The per-call seed is derived from self._seed XOR (T * 997) so that
    demos with different lengths get different frames while the selection
    is fully reproducible across runs for a fixed seed.
    """

    def __init__(self, n_keyframes: int, seed: int = 42) -> None:
        """
        Args:
            n_keyframes: Total number of keyframes to select (including
                         the mandatory first and last frames). Must be >= 2.
            seed:        Base random seed. Use the same value across conditions
                         for fair comparisons; change it for sensitivity checks.
        """
        if n_keyframes < 2:
            raise ValueError("n_keyframes must be >= 2 to include both endpoints.")
        self._n = n_keyframes
        self._seed = seed

    @property
    def name(self) -> str:
        return f"random_{self._n}"

    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Randomly select n_keyframes indices from the trajectory.

        Args:
            trajectory: Array of shape (T, D). Only T (length) is used.

        Returns:
            Sorted integer array of n indices spanning [0, T-1].
        """
        T = len(trajectory)
        if T <= self._n:
            return np.arange(T, dtype=int)

        rng = np.random.default_rng(self._seed ^ (T * 997))
        middle = rng.choice(np.arange(1, T - 1, dtype=int), size=self._n - 2, replace=False)
        return np.sort(np.concatenate([[0], middle, [T - 1]])).astype(int)
