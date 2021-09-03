import argparse
import logging
import uuid

import pyaudio
from flask import Flask, Response, stream_with_context
from flask import g as app_context

logger = logging.getLogger(__name__)


class Mic:
    BUFFER = 8

    def __init__(self, rate, channels, frame_size):
        self.id = str(uuid.uuid4())[:6]

        self._rate = rate
        self._channels = channels
        self._frame_size = frame_size

        self._pyaudio = pyaudio.PyAudio()
        self._active = False
        self._start_time = None
        self._time = None

    @property
    def active(self):
        return self._active

    @property
    def time(self):
        return self._mic_time - self._start_time

    @property
    def _mic_time(self):
        return self._time

    @_mic_time.setter
    def _mic_time(self, stream_time):
        advanced = stream_time - self._time
        if advanced > self._stream.get_input_latency():
            logger.exception("Latency exceeded buffer (%.4fsec) - dropped frames: %.4fsec",
                             self._stream.get_input_latency(), advanced)
        self._time = stream_time

    def stop(self):
        self._active = False
        logger.debug("Stopped microphone (%s)", self.id)

    def __enter__(self):
        self._stream = self._pyaudio.open(self._rate, self._channels, pyaudio.paInt16, input=True,
                                          frames_per_buffer=self.BUFFER * self._frame_size)
        self._active = True
        self._start_time = self._stream.get_time()
        self._time = self._start_time

        logger.debug("Opened microphone (%s) with rate: %s, channels: %s, frame_size: %s",
                     self.id, self._rate, self._channels, self._frame_size)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._active:
            self._active = False
            self._stream.close()
            logger.debug("Closed microphone (%s)", self.id)
        else:
            logger.warning("Ignored close microphone (%s)", self.id)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._active:
            raise StopIteration()

        data = self._stream.read(self._frame_size * self._channels, exception_on_overflow=False)
        self._mic_time = self._stream.get_time()

        return data


def backend_app(sampling_rate, channels, frame_size):
    app = Flask(__name__)

    @app.route('/mic')
    def stream_mic():
        mic = Mic(sampling_rate, channels, frame_size)

        def audio_stream(mic):
            with mic as mic_stream:
                yield from mic_stream

        # Store mic in (thread-local) app-context to be able to close it.
        app_context.mic = mic

        mime_type = f"audio/L16; rate={sampling_rate}; channels={channels}; frame_size={frame_size}"
        # mime_type = "text/plain"
        stream = stream_with_context(audio_stream(mic))

        return Response(stream, mimetype=mime_type)

    @app.after_request
    def set_cache_control(response):
      response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
      response.headers['Pragma'] = 'no-cache'
      response.headers['Expires'] = '0'

      return response

    @app.teardown_request
    def close_mic(_=None):
        if "mic" in app_context:
            app_context.mic.stop()

    return app


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='EMISSOR data processing')
    parser.add_argument('--rate', type=int, choices=[16000, 32000, 44100], default=16000, help="Sampling rate.")
    parser.add_argument('--channels', type=int, choices=[1, 2], default=2, help="Number of audio channels.")
    parser.add_argument('--frame_duration', type=int, choices=[10, 20, 30], default=30, help="Duration of audio frames in milliseconds.")
    parser.add_argument('--port', type=int, default=8000, help="Web server port")
    args, _ = parser.parse_known_args()

    logger.info("Starting webserver with args: %s", args)

    backend_app(args.rate, args.channels, args.frame_duration * args.rate // 1000).run(host="0.0.0.0", port=args.port)