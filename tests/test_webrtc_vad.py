import unittest

import numpy as np
import soundfile as sf
from importlib_resources import path
from parameterized import parameterized

from cltl.vad.webrtc_vad import WebRtcVAD

FRAME_FORMATS = [
    [10, 16000, None],
    [10, 16000, 1],
    [10, 16000, 2],
    [20, 16000, None],
    [20, 16000, 1],
    [20, 16000, 2],
    [30, 16000, None],
    [30, 16000, 1],
    [30, 16000, 2],
]


class TestVADUtil(unittest.TestCase):
    def setUp(self) -> None:
        self.vad = WebRtcVAD(mode=2, padding=0)

        # Calibrate VAD
        noise_array = np.random.randint(0, 50, (480,), dtype=np.int16)
        self.vad.is_vad(noise_array * 100, sampling_rate=16000)
        try:
            self.test_detect_vad()
        except:
            pass

    @parameterized.expand(FRAME_FORMATS)
    def test_is_vad_silence(self, duration, sampling_rate, channels):
        samples = sampling_rate * duration // 1000
        shape = (samples, channels) if channels else (samples)

        self.assertFalse(self.vad.is_vad(np.zeros(shape, dtype=np.int16), sampling_rate))

    def test_is_vad_invalid_frame_duration(self):
        with self.assertRaises(ValueError):
            samples = 16000 * 5 // 1000

            self.vad.is_vad(np.zeros((samples, 1), dtype=np.int16), 16000)

    def test_is_vad_invalid_sampling_rate(self):
        with self.assertRaises(NotImplementedError):
            samples = 32000 * 10 // 1000

            self.vad.is_vad(np.zeros((samples, 1), dtype=np.int16), 32000)

    @parameterized.expand(FRAME_FORMATS)
    def test_detect_vad_silence(self, duration, sampling_rate, channels):
        samples = sampling_rate * duration // 1000
        shape = (samples, channels) if channels else (samples)

        audio_frames = [np.zeros(shape, dtype=np.int16) for _ in range(10)]

        speech, _, _ = self.vad.detect_vad(audio_frames, sampling_rate)

        self.assertEqual(0, len(list(speech)))

    def test_detect_vad(self):
        with path("resources", "test.wav") as wav:
            speech_array, sampling_rate = sf.read(wav, dtype=np.int16)
            self.assertEqual(1, speech_array.ndim)

        duration = 30

        frame_length = (duration * sampling_rate) // 1000
        frames = len(speech_array) // frame_length
        audio_frames = np.split(speech_array[:frames * frame_length], frames)

        speech, _, _ = self.vad.detect_vad(audio_frames, 16000)
        speech = list(speech)

        self.assertEqual(62, len(speech))

    def test_detect_vad_skips_inactive(self):
        with path("resources", "test.wav") as wav:
            speech_array, sampling_rate = sf.read(wav, dtype=np.int16)
            self.assertEqual(1, speech_array.ndim)

        duration = 30

        frame_length = (duration * sampling_rate) // 1000
        frames = len(speech_array) // frame_length

        # Reduce noise to -20db and truncate speech to frame size
        noise_array = np.random.randint(0, int(0.01 * np.amax(speech_array)), (25 * frame_length,), dtype=np.int16)
        audio_array = np.concatenate([noise_array, speech_array[:frames * frame_length]])
        audio_frames = np.split(audio_array, 25 + frames)

        speech, _, _ = self.vad.detect_vad(audio_frames, 16000)
        speech = list(speech)

        self.assertEqual(62, len(speech))

    def test_detect_vad_terminates_at_speech_end(self):
        with path("resources", "test.wav") as wav:
            speech_array, sampling_rate = sf.read(wav, dtype=np.int16)
            self.assertEqual(1, speech_array.ndim)

        duration = 30
        sampling_rate = 16000

        frame_length = (duration * sampling_rate) // 1000
        frames = len(speech_array) // frame_length

        # Reduce noise to -20db and truncate speech to frame size
        noise_array = np.random.randint(0, int(0.01 * np.amax(speech_array)), (25 * frame_length,), dtype=np.int16)
        audio_array = np.concatenate([speech_array[:frames * frame_length], noise_array])
        audio_frames = np.split(audio_array, 25 + frames)

        audio_frames_iterator = iter(audio_frames)

        speech, _, _ = self.vad.detect_vad(audio_frames_iterator, 16000)
        speech = list(speech)

        self.assertEqual(62, len(speech))
        self.assertLess(0, len(list(audio_frames_iterator)))


    def _play(self, audio_array, sampling_rate):
        import sounddevice as sd
        sd.play(audio_array, sampling_rate)
        sd.wait()
