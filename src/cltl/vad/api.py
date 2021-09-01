import abc
from queue import Queue
from typing import Iterable

import numpy as np

class VadTimeout(Exception):
    def __init__(self, timeout):
        super().__init__(f"No voice activity within timeout ({timeout})")


class VAD(abc.ABC):
    def is_vad(self, audio_frame: np.array, sampling_rate: int) -> bool:
        raise NotImplementedError("")


    def detect_vad(self,
                   audio_frames: Iterable[np.array],
                   sampling_rate: int,
                   blocking: bool = True,
                   timeout: int = 0) -> Iterable[np.array]:
        """
        WIP

        Parameters
        ----------
        audio_frames : Iterable[np.array]
            Stream of audio frames on which voice activity will be detected.
            Implementations may support only specific frame formats.

        blocking : bool
            If True, the method blocks until voice activity is detected.

        timeout : int
            Maximum number of frames accepted for voice activity detection.


        Returns
        -------
        Queue[np.array]
            A contiguous section of audio frames with voice activity.
            If blocking is set to False, the returned Iterable will be threadsafe.

        Raises
        ------
        ValueError
            If the format of the provided audio_frames is not supported.

        VadTimeout
            If no voice activity was detected within the specified timeout.
        """
        raise NotImplementedError("")
