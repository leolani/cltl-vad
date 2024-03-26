import logging

import numpy as np
import webrtcvad

from cltl.vad.frame_vad import FrameWiseVAD

logger = logging.getLogger(__name__)


SAMPLING_RATES = {8000, 16000, 32000, 48000}
FRAME_DURATON = {10, 20, 30}
SAMPLE_DEPTH = {np.int16}


class WebRtcVAD(FrameWiseVAD):
    def __init__(self, activity_window: int = 1, activity_threshold: float = 1,
                 allow_gap: int = 0, padding: int = 2, min_duration: int = 0,
                 mode: int = 3, storage: str = None):
        logger.info("Setup WebRtcVAD with mode %s", mode)
        super().__init__(activity_window, activity_threshold, allow_gap, padding, min_duration, mode, storage)
        self._vad = webrtcvad.Vad(mode)

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