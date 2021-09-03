import logging
from queue import Queue
from typing import Iterable

import numpy as np
import webrtcvad

from cltl.vad.api import VAD
from cltl.vad.util import as_iterable

logger = logging.getLogger(__name__)


SAMPLING_RATES = set([8000, 16000, 32000, 48000])
FRAME_DURATON = set([10, 20, 30])
SAMPLE_DEPTH = set([np.int16])


class WebRtcVAD(VAD):
    def __init__(self, allow_gap: int = 0, mode: int = 3):
        self._vad = webrtcvad.Vad(mode)
        self._allow_gap = allow_gap

    def is_vad(self, audio_frame: np.array, sampling_rate: int) -> bool:
        if not audio_frame.dtype == np.int16:
            raise ValueError(f"Invalid sample depth {audio_frame.dtype}, expected np.int16")

        if sampling_rate != 16000:
            raise NotImplementedError(f"Currently only sampling rate 16000 is supported, was {sampling_rate}")

        frame_duration = (len(audio_frame) * 1000) // sampling_rate
        if not frame_duration in FRAME_DURATON:
            raise ValueError(f"Unsupported frame length {audio_frame.shape}, "
                             f"expected one of {[d * sampling_rate // 1000 for d in FRAME_DURATON]}ms "
                             f"(rate: {sampling_rate})")

        is_mono = audio_frame.ndim == 1 or audio_frame.shape[1] == 1
        mono_frame = audio_frame if is_mono else audio_frame.mean(axis=1).ravel()

        return self._vad.is_speech(mono_frame.tobytes(), sampling_rate, len(mono_frame))

    def detect_vad(self,
                   audio_frames: Iterable[np.array],
                   sampling_rate: int,
                   blocking: bool = True,
                   timeout: int = 0) -> Iterable[np.array]:
        if not blocking:
            raise NotImplementedError("Currently only blocking is supported")

        queue = Queue()

        offset = -1
        gap = None
        frame_duration = None
        for cnt, frame in enumerate(audio_frames):
            if cnt % 1000 == 0:
                logger.debug("Processing frame (%s)", cnt)
            if not frame_duration:
                frame_duration = len(frame) * 1000 / sampling_rate

            if self.is_vad(frame, sampling_rate):
                if offset < 0:
                    offset = cnt
                if gap:
                    list(map(queue.put, gap))
                gap = []
                queue.put(frame)
            elif gap and len(gap) * frame_duration > self._allow_gap:
                gap = None
                break
            elif gap is not None:
                gap.append(frame)

        queue.put(None)

        return as_iterable(queue), offset, cnt + 1