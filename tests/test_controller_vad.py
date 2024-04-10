import logging
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Iterable

import numpy as np
import time

from cltl.vad.api import VAD
from cltl.vad.controller_vad import ControllerVAD
from tests_integration.test_vad_service import TestVAD

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


SAMPLING_RATE = 16000
FRAME_DURATION = 30
FRAME_LENGTH = (FRAME_DURATION * SAMPLING_RATE) // 1000

PADDING = 10 * FRAME_DURATION


class TestVAD(VAD):
    def detect_vad(self, audio_frames: Iterable[np.ndarray], sampling_rate: int, blocking: bool = True,
                   timeout: int = 0) -> [Iterable[np.ndarray], int, int]:
        raise NotImplementedError()

    def is_vad(self, audio_frame: np.ndarray, sampling_rate: int) -> bool:
        return np.amax(audio_frame) == 1


def stop_after(silence: int, vad_length: int, vad: ControllerVAD, stop_latch: Event):
    cnt = 0
    for _ in range(silence):
        time.sleep(0.001)
        cnt += 1
        yield np.zeros((1,1))

    for _ in range(vad_length):
        time.sleep(0.001)
        cnt += 1
        yield np.ones((1,1))


    vad.active = False
    time.sleep(0.001)

    while not stop_latch.is_set():
        time.sleep(0.001)
        cnt += 1
        yield np.zeros((1,1))


class TestVADUtil(unittest.TestCase):
    def test_controller_vad(self):
        self.vad = ControllerVAD(TestVAD(), 3, min_duration=0)
        self.vad.active = True

        stop_latch = Event()
        executor = ThreadPoolExecutor(max_workers=1)
        result = executor.submit(lambda: self.vad.detect_vad(stop_after(10, 10, self.vad, stop_latch=stop_latch), 16000))
        time.sleep(0.1)
        stop_latch.set()
        audio, offset, consumed = result.result()
        audio = list(audio)

        self.assertLessEqual(23, consumed)
        self.assertEquals(7, offset)
        self.assertEquals(16, len(audio))

    def test_controller_vad_no_padding(self):
        self.vad = ControllerVAD(TestVAD(), 0, min_duration=0)
        self.vad.active = True

        stop_latch = Event()
        executor = ThreadPoolExecutor(max_workers=1)
        result = executor.submit(lambda: self.vad.detect_vad(stop_after(10, 10, self.vad, stop_latch=stop_latch), 16000))
        time.sleep(0.1)
        stop_latch.set()
        audio, offset, consumed = result.result()
        audio = list(audio)

        self.assertLessEqual(20, consumed)
        self.assertEquals(10, offset)
        self.assertEquals(10, len(audio))

    def test_controller_vad_no_silence(self):
        self.vad = ControllerVAD(TestVAD(), 0, min_duration=0)
        self.vad.active = True

        stop_latch = Event()
        executor = ThreadPoolExecutor(max_workers=1)
        result = executor.submit(lambda: self.vad.detect_vad(stop_after(0, 10, self.vad, stop_latch=stop_latch), 16000))
        time.sleep(0.1)
        stop_latch.set()
        audio, offset, consumed = result.result()
        audio = list(audio)

        self.assertLessEqual(10, consumed)
        self.assertEquals(0, offset)
        self.assertEquals(10, len(audio))

    def test_controller_vad_silence_less_than_padding(self):
        self.vad = ControllerVAD(TestVAD(), 10, min_duration=0)
        self.vad.active = True

        stop_latch = Event()
        executor = ThreadPoolExecutor(max_workers=1)
        result = executor.submit(lambda: self.vad.detect_vad(stop_after(5, 10, self.vad, stop_latch=stop_latch), 16000))
        time.sleep(0.1)
        stop_latch.set()
        audio, offset, consumed = result.result()
        audio = list(audio)

        self.assertLessEqual(25, consumed)
        self.assertEquals(0, offset)
        self.assertEquals(25, len(audio))
