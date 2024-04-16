import logging
from typing import Callable

import flask
from cltl.backend.source.client_source import ClientAudioSource
from cltl.backend.spi.audio import AudioSource
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl_service.vad.schema import VadMentionEvent
from flask import Response

from cltl.vad.controller_vad import ControllerVAD
from cltl_service.vad.service import VadService

logger = logging.getLogger(__name__)


class ControllerVadService(VadService):
    @classmethod
    def from_ctrl_config(cls, vad: ControllerVAD, event_bus: EventBus, resource_manager: ResourceManager,
                         config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.vad")
        ctrl_config = config_manager.get_config("cltl.vad.controller")

        def audio_loader(url, offset, length) -> AudioSource:
            return ClientAudioSource.from_config(config_manager, url, offset, length)

        return cls(ctrl_config.get("control_topic"), config.get("mic_topic"), config.get("vad_topic"),
                   vad, audio_loader, event_bus, resource_manager)

    def __init__(self, control_topic: str, mic_topic: str, vad_topic: str,
                 vad: ControllerVAD, audio_loader: Callable[[str, int, int], AudioSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        super().__init__(mic_topic, vad_topic, vad, audio_loader, event_bus, resource_manager)
        self._control_topic = control_topic

        self._app = None

    @property
    def input_topics(self):
        return super().input_topics + [self._control_topic]

    def _process(self, event: Event):
        if event.metadata.topic == self._control_topic:
            logger.debug("Controller VAD %s", "activated" if event.payload else "deactivated")
            self._vad.active = event.payload
        else:
            super()._process(event)

    def _vad_task(self, payload):
        audio_id, url = (payload.signal.id, payload.signal.files[0])

        def detect():
            consumed = -1
            source_offset = 0
            while not self._stopped.value and consumed != 0:
                speech, offset, consumed, frame_size = self._listen(url, source_offset)
                speech = list(speech)

                vad_event = None
                if len(speech) > 0:
                    speech_offset = source_offset + (offset * frame_size)
                    vad_event = self._create_payload(speech, speech_offset, payload)
                    logger.debug("Published VAD event (offset: %s, consumed %s)", offset, consumed)
                elif consumed != 0 and offset >= 0:
                    # Don't send an event if no VAD was detected before audio ends
                    vad_event = VadMentionEvent(VadMentionEvent.__name__, [])

                if vad_event:
                    self._event_bus.publish(self._vad_topic, Event.for_payload(vad_event))

                source_offset += consumed * frame_size

        return detect

    @property
    def app(self):
        if self._app:
            return self._app

        self._app = flask.Flask(__name__)

        @self._app.route('/rest/active', methods=['GET', 'POST'])
        def voice_activity():
            if flask.request.method == 'GET':
                return str(self._vad.active)
            if flask.request.method == 'POST':
                self._vad.active = True
                logger.debug("VAD activated via POST")

                return str(True)

        @self._app.route('/rest/stop', methods=['POST'])
        def stop_va():
            self._vad.active = False

            logger.debug("VAD deactivated")

            return Response(status=200)

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