import threading
import unittest
from queue import Queue, Empty
from typing import Iterable

import numpy as np
from cltl.backend.spi.audio import AudioSource
from cltl.combot.infra.event import Event
from cltl.combot.infra.event.memory import SynchronousEventBus
from cltl_service.backend.schema import AudioSignalStarted
from emissor.representation.scenario import AudioSignal

from cltl.vad.api import VAD
from cltl_service.vad.service import VadService


def wait(lock: threading.Event):
    if not lock.wait(1):
        raise unittest.TestCase.failureException("Latch timed out")


def test_source(start, speech_started, speech_ended):
    class TestSource(AudioSource):
        zeros = np.zeros((16, 1), dtype=np.int16)
        ones = np.ones((16, 1), dtype=np.int16)

        def __init__(self, url, offset, length):
            self.offset = offset // 16

        @property
        def audio(self) -> Iterable[np.array]:
            """
            Frames: [0,0,1,1,0,0,0,1,1,1,0]
            Expected start offsets: 0, 5
            Expected speech offsets: 2, 7
            """
            if self.offset > 9:
                return
            if self.offset != 0 and self.offset != 5:
                print("Failure:", self.offset)
                raise unittest.TestCase.failureException(self.offset)

            if self.offset == 0:
                yield from [self.zeros, self.zeros]

                wait(start)
                start.clear()
                yield self.ones

                speech_started.set()
                yield self.ones

                wait(start)
                speech_started.clear()
                speech_ended.set()
                start.clear()
                yield self.zeros

            wait(start)
            start.clear()
            yield from [self.zeros, self.zeros]
            yield self.ones

            speech_started.set()
            wait(start)
            yield from [self.ones, self.ones]
            yield self.zeros

        @property
        def rate(self):
            return 16000

        @property
        def channels(self):
            return 1

        @property
        def frame_size(self):
            return 16

        @property
        def depth(self):
            return 2

    return TestSource


class DummyVad(VAD):
    def is_vad(self, audio_frame: np.array, sampling_rate: int) -> bool:
        return audio_frame.sum() > 0

    def detect_vad(self, audio_frames: Iterable[np.array], sampling_rate: int, blocking: bool = True,
                   timeout: int = 0) -> [Iterable[np.array], int, int]:
        is_vad = False
        offset = 0
        speech = []
        for last, frame in enumerate(audio_frames):
            if is_vad and not self.is_vad(frame, sampling_rate):
                return speech, offset, last + 1
            if not is_vad and self.is_vad(frame, sampling_rate):
                is_vad = True
                offset = last
            if is_vad:
                speech.append(frame)

        return speech, offset, last + 1


class TestVAD(unittest.TestCase):
    def setUp(self) -> None:
        self.event_bus = SynchronousEventBus()
        self.vad_service = None

    def tearDown(self) -> None:
        if self.vad_service:
            self.vad_service.stop()

    def test_events_from_vad_service(self):
        start = threading.Event()
        speech_started = threading.Event()
        speech_ended = threading.Event()

        self.vad_service = VadService("mic_topic", "vad_topic", DummyVad(),
                                      test_source(start, speech_started, speech_ended), self.event_bus, None)
        self.vad_service.start()

        audio_signal = AudioSignal.for_scenario("scenario_id", 0, 1,
                                 f"cltl-storage:audio/1",
                                 1, 2, signal_id=1)
        audio_started = AudioSignalStarted.create(audio_signal)
        self.event_bus.publish("mic_topic", Event.for_payload(audio_started))

        events = Queue()

        def receive_event(event):
            events.put(event)

        self.event_bus.subscribe("vad_topic", receive_event)

        start.set()
        wait(speech_started)

        with self.assertRaises(Empty):
            events.get(block=True, timeout=0.1)

        start.set()
        wait(speech_ended)

        event = events.get(block=True, timeout=0.1)
        self.assertEqual(2 * 16, event.payload.mentions[0].segment[0].start)
        self.assertEqual(4 * 16, event.payload.mentions[0].segment[0].stop)

        start.set()
        wait(speech_started)

        with self.assertRaises(Empty):
            events.get(block=True, timeout=0.1)

        start.set()

        event = events.get(block=True, timeout=0.1)
        self.assertEqual(7 * 16, event.payload.mentions[0].segment[0].start)
        self.assertEqual(10 * 16, event.payload.mentions[0].segment[0].stop)
