from abc import ABC, abstractmethod

import numpy as np


class KeyframeExtractor(ABC):
    """Abstract base class for all keyframe extraction methods."""

    @abstractmethod
    def extract(self, trajectory: np.ndarray) -> np.ndarray:
        """Select keyframe indices from a trajectory array.

        Args:
            trajectory: Array of shape (T, …). May be a 1-D signal (T,),
                        a 2-D feature array (T, D), or a 4-D frame stack
                        (T, H, W, C) — subclass decides what it reads.

        Returns:
            1-D integer array of selected frame indices, sorted ascending.
            Always includes index 0 and T-1.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier used in plots and result filenames."""

    @staticmethod
    def compression_ratio(original_len: int, selected_len: int) -> float:
        """Fraction of frames retained (lower = more compressed).

        Args:
            original_len: Total number of frames in the trajectory.
            selected_len: Number of selected keyframes.

        Returns:
            selected_len / original_len, or 0.0 if original_len is 0.
        """
        if original_len == 0:
            return 0.0
        return selected_len / original_len
