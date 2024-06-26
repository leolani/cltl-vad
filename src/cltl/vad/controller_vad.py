import logging
from collections import deque
from queue import Queue
from threading import Event
from typing import Iterable

import numpy as np

from cltl.vad.api import VAD
from cltl.vad.util import as_iterable

logger = logging.getLogger(__name__)


class ControllerVAD(VAD):
    def __init__(self, vad: VAD, padding_size: int, min_duration: int):
        self._vad = vad
        self._active = Event()
        self._padding = padding_size

    @property
    def active(self) -> bool:
        return self._active.is_set()

    @active.setter
    def active(self, is_active):
        if is_active:
            self._active.set()
            logger.debug("VA set active")
        else:
            logger.debug("VA set inactive")
            self._active.clear()

    def is_vad(self, audio_frame: np.ndarray, sampling_rate: int) -> bool:
        return self.active

    def detect_vad(self, audio_frames: Iterable[np.ndarray], sampling_rate: int, blocking: bool = True,
                   timeout: int = 0) -> [Iterable[np.ndarray], int, int]:
        audio_iter = iter(audio_frames)

        try:
            frame = next(audio_iter)
            cnt = 1
        except StopIteration:
            logger.debug("Empty audio in VAD")
            return [], -1, 0

        frame_duration = 1000 * len(frame) / sampling_rate
        padding_size = int(self._padding // frame_duration)

        padding_buffer = deque(maxlen=padding_size)
        audio = []

        while not self.active:
            try:
                # Drop audio if not active
                frame = next(audio_iter)
                cnt += 1
            except StopIteration:
                logger.debug("No VA in controlled audio of length %s", cnt)
                return [], -1, cnt

        while self.active and not self._vad.is_vad(frame, sampling_rate):
            try:
                padding_buffer.append(frame)
                frame = next(audio_iter)
                cnt += 1
            except StopIteration:
                logger.debug("No VA in controlled audio of length %s", cnt)
                return [], -1, cnt

        offset = cnt - len(padding_buffer) - 1
        list(map(audio.append, padding_buffer))
        logger.debug("Detected start of VA at offset %s cnt (padding %s)", offset, len(padding_buffer))

        while self.is_vad(frame, sampling_rate):
            try:
                audio.append(frame)
                frame = next(audio_iter)
                cnt += 1
            except StopIteration:
                logger.debug("Detected VA of length: %s", len(audio))
                return tuple(audio), offset, cnt

        for _ in range(padding_size):
            try:
                audio.append(frame)
                frame = next(audio_iter)
                cnt += 1
            except:
                break

        logger.debug("Detected VA of length: %s", len(audio))

        return tuple(audio), offset, cnt
