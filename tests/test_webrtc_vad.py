import logging
import unittest

import numpy as np
import soundfile as sf
from importlib.resources import path
from parameterized import parameterized

from cltl.vad.webrtc_vad import WebRtcVAD
from tests.test_util import plot_wav


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


SAMPLING_RATE = 16000
FRAME_DURATION = 30
FRAME_LENGTH = (FRAME_DURATION * SAMPLING_RATE) // 1000

PADDING = 10 * FRAME_DURATION
ACTIVITY_WINDOW = 4 * FRAME_DURATION
ALLOW_GAP = 10 * FRAME_DURATION
ACTIVITY_THRESHOLD = 0.7


FRAME_FORMATS = [
    # duration, channels
    [10, None],
    [10, 1],
    [10, 2],
    [20, None],
    [20, 1],
    [20, 2],
    [30, None],
    [30, 1],
    [30, 2],
]


TEST_SPEECH = [
    # name, file, offset (ms), gap (ms), length (ms)
    ["short, no gap", "short_155.wav", 1000, 0, 155],
    ["short, short gap", "short_155.wav", 1000, 150, 155],
    ["short, gap", "short_155.wav", 1000, 290, 155],
    ["short, large gap", "short_155.wav", 1000, 450, 155],
    ["short, no silence", "short_155.wav", 10, 0, 155],
    ["short, no silence, short gap", "short_155.wav", 10, 150, 155],
    ["short, no silence, gap", "short_155.wav", 10, 290, 155],
    ["short, no silence, large gap", "short_155.wav", 10, 450, 155],
    ["long, no gap", "long_1460.wav", 1000, 0, 1460],
    ["long, short gap", "long_1460.wav", 1000, 150, 1460],
    ["long, gap", "long_1460.wav", 1000, 290, 1460],
    ["long, large gap", "long_1460.wav", 1000, 450, 1460],
    ["long, no silence", "long_1460.wav", 10, 0, 1460],
    ["long, no silence, short gap", "long_1460.wav", 10, 150, 1460],
    ["long, no silence, gap", "long_1460.wav", 10, 290, 1460],
    ["long, no silence, large gap", "long_1460.wav", 10, 450, 1460],
]


class TestVADUtil(unittest.TestCase):
    def setUp(self) -> None:
        self.vad = WebRtcVAD(activity_window=ACTIVITY_WINDOW, activity_threshold=ACTIVITY_THRESHOLD, allow_gap=ALLOW_GAP,
                             min_duration=90, mode=2, padding=PADDING)

        # Calibrate VAD
        noise_array = np.random.randint(0, 50, (480,), dtype=np.int16)
        self.vad.is_vad(noise_array * 100, sampling_rate=SAMPLING_RATE)
        try:
            self.test_detect_vad()
        except:
            pass

    @parameterized.expand(FRAME_FORMATS)
    def test_is_vad_silence(self, duration, channels):
        samples = SAMPLING_RATE * duration // 1000
        shape = (samples, channels) if channels else (samples)

        self.assertFalse(self.vad.is_vad(np.zeros(shape, dtype=np.int16), SAMPLING_RATE))

    def test_is_vad_invalid_frame_duration(self):
        with self.assertRaises(ValueError):
            samples = SAMPLING_RATE * 5 // 1000

            self.vad.is_vad(np.zeros((samples, 1), dtype=np.int16), SAMPLING_RATE)

    def test_is_vad_invalid_sampling_rate(self):
        with self.assertRaises(NotImplementedError):
            samples = 32000 * 10 // 1000

            self.vad.is_vad(np.zeros((samples, 1), dtype=np.int16), 32000)

    @parameterized.expand(FRAME_FORMATS)
    def test_detect_vad_silence(self, duration, channels):
        samples = SAMPLING_RATE * duration // 1000
        shape = (samples, channels) if channels else (samples)

        audio_frames = [np.zeros(shape, dtype=np.int16) for _ in range(10)]

        speech, _, _ = self.vad.detect_vad(audio_frames, SAMPLING_RATE)

        self.assertEqual(0, len(list(speech)))

    def test_detect_vad(self):
        with path("resources", "test.wav") as wav:
            speech_array, sampling_rate = sf.read(wav, dtype=np.int16)
            self.assertEqual(1, speech_array.ndim)

        duration = 30

        frame_length = (duration * sampling_rate) // 1000
        frames = len(speech_array) // frame_length
        audio_frames = np.split(speech_array[:frames * frame_length], frames)

        speech, _, _ = self.vad.detect_vad(audio_frames, SAMPLING_RATE)
        speech = list(speech)

        self.assertEqual(69, len(speech))

    @parameterized.expand(TEST_SPEECH)
    def test_detect_vad_with_paramters(self, _, file, offset, gap, length):
        self.detect_vad_with_parameters(_, file, offset, gap, length)

    # Debug parameterized
    def test_detect_vad_single_file(self):
        self.detect_vad_with_parameters("", "long_1460.wav", 1000, 450, 1460)

    def detect_vad_with_parameters(self, _, file, offset, gap, length):
        with path("resources", file) as wav:
            speech_array, sampling_rate = sf.read(wav, dtype=np.int16)
            self.assertEqual(1, speech_array.ndim)

        # add start
        audio_array = self.add_noise(speech_array, offset, start=True)
        # add gap
        if gap:
            audio_array = self.add_noise(audio_array, gap, start=False)
            audio_array = np.concatenate([audio_array, speech_array])
        # add end
        audio_array = self.add_noise(audio_array, 1000, start=False)

        total_frames = len(audio_array) // FRAME_LENGTH
        audio_frames = np.split(audio_array[:total_frames * FRAME_LENGTH], total_frames)
        audio_frames_iterator = iter(audio_frames)

        speech, actual_offset, acutal_consumed = self.vad.detect_vad(audio_frames_iterator, SAMPLING_RATE)
        speech = list(speech)

        # Debug
        # self.plot(audio_frames, [actual_offset, actual_offset + len(speech)])

        expected_frames = (min(PADDING, offset) + length + (gap + length if 0 < gap and gap <= ALLOW_GAP else 0) + PADDING) // FRAME_DURATION
        expected_offset = max(0, offset - PADDING) // FRAME_DURATION
        tolerance = ACTIVITY_WINDOW // FRAME_DURATION + 2

        self.assertAlmostEquals(expected_frames, len(speech), delta=tolerance)
        self.assertAlmostEquals(expected_offset, actual_offset, delta=tolerance)
        self.assertGreaterEqual(acutal_consumed, actual_offset + len(speech) + 1)

    def add_noise(self, speech_array, duration, start=True):
        if duration == 0:
            return speech_array

        # Reduce noise to -20db and truncate speech to frame size
        level = int(0.01 * np.amax(speech_array))
        samples = (duration * SAMPLING_RATE) // 1000
        noise_array = np.random.randint(0, level, samples, dtype=np.int16)

        reverse = int(start) * 2 - 1

        return np.concatenate([noise_array, speech_array][::reverse])

    # Debug helpers
    def play(self, audio_array, sampling_rate):
        import sounddevice as sd
        sd.play(audio_array, sampling_rate)
        sd.wait()

    def plot(self, audio_frames, marked_frames):
        plot_wav(np.concatenate(audio_frames), SAMPLING_RATE, FRAME_DURATION, [f * FRAME_LENGTH for f in marked_frames])
