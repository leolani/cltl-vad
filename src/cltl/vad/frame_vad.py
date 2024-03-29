import abc
import logging
from collections import deque
from itertools import chain, islice
from queue import Queue

import numpy as np
from cltl.combot.infra.time_util import timestamp_now
from typing import Iterable

from cltl.vad.api import VAD, VadTimeout
from cltl.vad.util import as_iterable, store_frames

logger = logging.getLogger(__name__)


class FrameWiseVAD(VAD, abc.ABC):
    def __init__(self, activity_window: int = 1, activity_threshold: float = 1,
                 allow_gap: int = 0, padding: int = 2, min_duration: int = 0,
                 mode: int = 3, storage: str = None):
        logger.info("Setup WebRtcVAD with mode %s", mode)
        self._activity_window = activity_window
        self._activity_threshold = activity_threshold
        self._allow_gap = allow_gap
        self._padding = padding
        self._min_duration = min_duration
        self._storage = storage

    def detect_vad(self,
                   audio_frames: Iterable[np.array],
                   sampling_rate: int,
                   blocking: bool = True,
                   timeout: int = 0) -> Iterable[np.array]:
        if not blocking:
            raise NotImplementedError("Currently only blocking is supported")

        storage_buffer = []

        audio_frames = iter(audio_frames)
        try:
            first = next(audio_frames)
        except StopIteration:
            return [], -1, 0

        frame_duration = 1000 * len(first) / sampling_rate
        window_size = max(1, int(self._activity_window // frame_duration))
        padding_size = int(self._padding // frame_duration)
        gap_size = int(self._allow_gap // frame_duration)
        padding_buffer = deque(maxlen=padding_size + window_size - 1)

        voice_activity = Queue()

        # Initialized during processing
        offset = -1
        gap = None
        va_length = 0

        logger.debug("Started VAD with window of %s and padding of %s frames (%s ms frame duration)",
                     window_size, padding_size, frame_duration)

        for cnt, frame, activity in self._with_average_activity(chain((first,), audio_frames), sampling_rate, window_size):
            storage_buffer.append(frame)
            if offset < 0 and timeout > 0 and self._cnt_to_sec(cnt, frame_duration) > timeout:
                raise VadTimeout(f"No VA detected within timeout ({timeout})")

            # Debug
            # if cnt % 100 == 0:
            #     logger.debug("Processing frames (%s - %sms) : %s", cnt, cnt * frame_duration, to_decibel(storage_buffer[cnt-100:cnt]))

            if activity and activity >= self._activity_threshold:
                if voice_activity.qsize() == 0:
                    padding = list(islice(padding_buffer, padding_size))
                    offset = cnt - len(padding)
                    logger.debug("Detected start of VA at %s, set offset to %s (padding: %s) frames", cnt, offset, len(padding))
                    list(map(voice_activity.put, padding))
                    padding_buffer = deque(maxlen=padding_size + window_size - 1)
                if gap:
                    logger.debug("Detected gap of %s in VA at %s", len(gap), cnt)
                    list(map(voice_activity.put, gap))
                gap = []
                voice_activity.put(frame)
                va_length += 1
            elif gap and len(gap) * frame_duration > self._allow_gap:
                if va_length * frame_duration >= self._min_duration:
                    logger.debug("Detected end of VA at %s, start padding", cnt)
                    break
                else:
                    logger.debug("Reset VA detection for short VA of %s", va_length)
                    voice_activity = Queue()
                    va_length = 0
                    gap = None
            elif gap is not None:
                gap.append(frame)
            else:
                padding_buffer.append(frame)

        if gap:
            list(map(voice_activity.put, islice(gap, padding_size)))

        try:
            for _ in range(max(0, padding_size - gap_size)):
                voice_activity.put(next(iter(audio_frames)))
                cnt += 1
        except StopIteration:
            logger.debug("Reached end of audio at %s", cnt)
            pass

        voice_activity.put(None)

        logger.debug("Detected VA of length: %s", voice_activity.qsize() - 1)
        if self._storage:
            key = f"{int(timestamp_now())}-{offset}"
            store_frames(storage_buffer, sampling_rate, save=f"{self._storage}/vad-{key}.wav")

        return as_iterable(voice_activity), offset, cnt + 1

    def _cnt_to_sec(self, cnt, frame_duration):
        if frame_duration is None:
            return 0

        return cnt * frame_duration // 1000

    # From https://docs.python.org/3/library/collections.html#deque-recipes
    def _with_average_activity(self, audio_frames, sampling_rate, size):
        it = enumerate(audio_frames)
        head = list(islice(it, size - 1))

        window = deque(int(self.is_vad(f, sampling_rate)) for i, f in head)
        window.appendleft(0)
        total = sum(window)

        for cnt, frame in head:
            # TODO None??
            yield cnt, frame, total / float(size)

        for cnt, frame in it:
            is_vad = int(self.is_vad(frame, sampling_rate))
            total += is_vad - window.popleft()
            window.append(is_vad)
            yield cnt, frame, total / float(size)