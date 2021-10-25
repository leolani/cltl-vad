import abc
from typing import Iterable

import numpy as np


class VadTimeout(Exception):
    def __init__(self, timeout):
        super().__init__(f"No voice activity within timeout ({timeout})")


class VAD(abc.ABC):
    def is_vad(self, audio_frame: np.ndarray, sampling_rate: int) -> bool:
        raise NotImplementedError("")


    def detect_vad(self,
                   audio_frames: Iterable[np.ndarray],
                   sampling_rate: int,
                   blocking: bool = True,
                   timeout: int = 0) -> [Iterable[np.ndarray], int, int]:
        """
        WIP

        Parameters
        ----------
        audio_frames : Iterable[np.array]
            Stream of audio frames on which voice activity will be detected.
            Implementations may support only specific frame formats.

        sampling_rate : int
            The sampling rate of the audio frames

        blocking : bool
            If True, the method blocks until voice activity is detected.

        timeout : float
            Maximum duration of audio frames accepted for voice activity detection
            in seconds.

        Returns
        -------
        Iterable[np.array]
            A contiguous section of audio frames with voice activity.
            If blocking is set to False, the returned Iterable will be thread-safe.
        int
            The offset of the output frames in the input stream.
        int
            The number of frames consumed from the input stream.

        Raises
        ------
        ValueError
            If the format of the provided audio_frames is not supported.

        VadTimeout
            If no voice activity was detected within the specified timeout.
        """
        raise NotImplementedError("")
