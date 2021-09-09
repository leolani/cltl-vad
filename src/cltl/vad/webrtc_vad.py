import time

import logging
import numpy as np
import webrtcvad
from queue import Queue, Empty
from typing import Iterable

from cltl.vad.api import VAD, VadTimeout
from cltl.vad.util import as_iterable, store_frames, to_decibel

logger = logging.getLogger(__name__)


SAMPLING_RATES = set([8000, 16000, 32000, 48000])
FRAME_DURATON = set([10, 20, 30])
SAMPLE_DEPTH = set([np.int16])


class WebRtcVAD(VAD):
    def __init__(self, allow_gap: int = 0, padding: int = 2, mode: int = 3, storage: str = None):
        logger.info("Setup WebRtcVAD with mode %s", mode)
        self._vad = webrtcvad.Vad(mode)
        self._allow_gap = allow_gap
        self._padding = padding
        self._storage = storage

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
        mono_frame = audio_frame if is_mono else audio_frame.mean(axis=1, dtype=np.int16).ravel()

        return self._vad.is_speech(mono_frame.tobytes(), sampling_rate, len(mono_frame))

    def detect_vad(self,
                   audio_frames: Iterable[np.array],
                   sampling_rate: int,
                   blocking: bool = True,
                   timeout: int = 0) -> Iterable[np.array]:
        if not blocking:
            raise NotImplementedError("Currently only blocking is supported")

        queue = Queue()
        storage_buffer = []
        offset = -1
        gap = None
        frame_duration = None
        padding_buffer = Queue(self._padding) if self._padding else None
        audio_frames = iter(audio_frames)
        for cnt, frame in enumerate(audio_frames):
            storage_buffer.append(frame)
            if offset < 0 and timeout > 0 and self._cnt_to_sec(cnt, frame_duration) > timeout:
                raise VadTimeout(timeout)

            if not frame_duration:
                frame_duration = len(frame) * 1000 / sampling_rate

            # if cnt % 100 == 0:
            #     logger.debug("Processing frames (%s - %sms) : %s", cnt, cnt * frame_duration, to_decibel(storage_buffer[cnt-100:cnt]))

            if self.is_vad(frame, sampling_rate):
                if offset < 0:
                    offset = cnt
                    logger.debug("Detected VA at %s", offset)
                    if padding_buffer is not None:
                        [queue.put(f) for f in padding_buffer.queue]
                if gap:
                    list(map(queue.put, gap))
                gap = []
                queue.put(frame)
            elif gap and len(gap) * frame_duration > self._allow_gap:
                gap = None
                if queue.qsize() * frame_duration > 3 * self._allow_gap:
                    break
                else:
                    queue = Queue()
            elif gap is not None:
                gap.append(frame)

            if padding_buffer is not None:
                try:
                    padding_buffer.get_nowait()
                except Empty:
                    pass
                padding_buffer.put(frame, )

        try:
            for _ in range(self._padding):
                queue.put(next(iter(audio_frames)))
                cnt += 1
        except StopIteration:
            pass

        queue.put(None)

        logger.debug("Detected VA of length: %s", queue.qsize())
        if self._storage:
            key = f"{int(time.time())}-{offset}"
            store_frames(storage_buffer, sampling_rate, save=f"{self._storage}/vad-{key}.wav")

        return as_iterable(queue), offset, cnt + 1

    def _cnt_to_sec(self, cnt, frame_duration):
        if frame_duration is None:
            return 0

        return cnt * frame_duration // 1000
