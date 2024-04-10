import logging
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable

import flask
from cltl.backend.source.client_source import ClientAudioSource
from cltl.backend.spi.audio import AudioSource
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.combot.infra.util import ThreadsafeBoolean
from cltl.combot.event.emissor import AudioSignalStarted, AudioSignalStopped
from cltl_service.vad.service import VadService
from emissor.representation.container import Index

from cltl.vad.api import VAD
from cltl.vad.controller_vad import ControllerVAD
from cltl_service.vad.schema import VadAnnotation, VadMentionEvent

logger = logging.getLogger(__name__)


class ControllerVadService(VadService):
    @classmethod
    def from_config(cls, vad: ControllerVAD, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.vad")

        def audio_loader(url, offset, length) -> AudioSource:
            return ClientAudioSource.from_config(config_manager, url, offset, length)

        return cls(config.get("mic_topic"), config.get("vad_topic"), vad, audio_loader, event_bus, resource_manager)

    def __init__(self, mic_topic: str, vad_topic: str, vad: ControllerVAD, audio_loader: Callable[[str, int, int], AudioSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        super().__init__(mic_topic, vad_topic, vad, audio_loader, event_bus, resource_manager)

        self._app = None

    def _process(self, event):
        if event.payload.type == AudioSignalStarted.__name__:
            self._vad.active = True

        super()._process(event)

    @property
    def app(self):
        if self._app:
            return self._app

        self._app = flask.Flask(__name__)

        @self._app.route('/vad/active')
        def activate():
            return self._vad.active

        @self._app.route('/vad/stop', methods=['POST'])
        def activate():
            self._vad.active = False

        @self._app.route('/urlmap')
        def url_map():
            return str(self._app.url_map)

        @self._app.after_request
        def set_cache_control(response):
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'

            return response

        return self._app