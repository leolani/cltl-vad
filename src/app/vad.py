import logging
import time
from types import SimpleNamespace

import numpy as np
import requests
from flask import Flask, Response, request

from cltl.vad.api import VadTimeout
from cltl.vad.webrtc_vad import WebRtcVAD

logger = logging.getLogger(__name__)


CONTENT_TYPE_SEPARATOR = ';'


def _play(frames, sampling_rate, save=None):
    import sounddevice as sd
    audio = np.concatenate(frames)
    print("Playing:", audio.shape)
    sd.play(audio, sampling_rate)
    sd.wait()

    if save:
        import soundfile
        soundfile.write(save, audio, sampling_rate)


def vad_app(allow_gap=100, padding=2, mode=2, timeout=10, storage=None):
    app = Flask(__name__)

    vad = WebRtcVAD(allow_gap=allow_gap, padding=padding, mode=mode, storage=storage)

    # TODO remove duplication
    @app.route('/calibrate')
    def calibrate():
        url = request.args.get('url')
        duration = request.args.get('sec', default=10, type=int)

        with requests.get(url, stream=True) as source:
            content_type = source.headers['content-type'].split(CONTENT_TYPE_SEPARATOR)
            if not content_type[0].strip() == 'audio/L16' or len(content_type) != 4:
                # Only support 16bit audio for now
                raise ValueError("Unsupported content type {content_type[0]}, "
                                 "expected audio/L16 with rate, channels and frame_size paramters")

            parameters = SimpleNamespace(**{p.split('=')[0].strip(): int(p.split('=')[1].strip())
                                            for p in content_type[1:]})

            logger.debug("Listening to %s (%s, %s)", url, content_type[0], parameters)

            # Two bytes per sample for 16bit audio
            bytes_per_frame = parameters.frame_size * parameters.channels * 2
            content = (np.frombuffer(frame, np.int16).reshape((parameters.frame_size, parameters.channels))
                       for frame in source.iter_content(bytes_per_frame))

            start = time.time()
            while time.time() - start < duration:
                try:
                    speech, _, _ = vad.detect_vad(content, parameters.rate, blocking=True, timeout=10)
                except VadTimeout:
                    pass

        return Response(status=200)

    @app.route('/listen')
    def listen():
        url = request.args.get('url')

        with requests.get(url, stream=True) as source:
            content_type = source.headers['content-type'].split(CONTENT_TYPE_SEPARATOR)
            if not content_type[0].strip() == 'audio/L16' or len(content_type) != 4:
                # Only support 16bit audio for now
                raise ValueError("Unsupported content type {content_type[0]}, "
                                 "expected audio/L16 with rate, channels and frame_size paramters")

            parameters = SimpleNamespace(**{p.split('=')[0].strip(): int(p.split('=')[1].strip())
                                            for p in content_type[1:]})

            logger.debug("Listening to %s (%s, %s)", url, content_type[0], parameters)

            # Two bytes per sample for 16bit audio
            bytes_per_frame = parameters.frame_size * parameters.channels * 2
            content = (np.frombuffer(frame, np.int16).reshape((parameters.frame_size, parameters.channels))
                       for frame in source.iter_content(bytes_per_frame))

            try:
                speech, _, _ = vad.detect_vad(content, parameters.rate, blocking=True, timeout=timeout)
            except VadTimeout:
                return Response(status=400)

        return Response((frame.tobytes() for frame in speech), mimetype=source.headers['content-type'])

    @app.after_request
    def set_cache_control(response):
      response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
      response.headers['Pragma'] = 'no-cache'
      response.headers['Expires'] = '0'

      return response

    return app
