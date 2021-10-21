import logging
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.topic_worker import TopicWorker
from cltl.combot.infra.util import ThreadsafeBoolean
from concurrent.futures.thread import ThreadPoolExecutor
from emissor.representation.container import Index
from typing import Type, Callable

from cltl.backend.source.client_source import ClientAudioSource
from cltl.backend.spi.audio import AudioSource
from cltl.vad.api import VAD
from cltl_service.backend.schema import AudioSignalStarted, AudioSignalStopped
from cltl_service.vad.schema import VadAnnotation, VadMentionEvent

logger = logging.getLogger(__name__)


CONTENT_TYPE_SEPARATOR = ';'


class VadService:
    @classmethod
    def from_config(cls, vad: VAD, event_bus: EventBus, resource_manager: ResourceManager,
                    config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.vad")

        def audio_loader(url, offset, length) -> AudioSource:
            return ClientAudioSource.from_config(config_manager, url, offset, length)

        return cls(config.get("mic_topic"), config.get("vad_topic"), vad, audio_loader, event_bus, resource_manager)

    def __init__(self, mic_topic: str, vad_topic: str, vad: VAD, audio_loader: Callable[[str, int, int], AudioSource],
                 event_bus: EventBus, resource_manager: ResourceManager):
        self._vad = vad
        self._audio_loader = audio_loader
        self._event_bus = event_bus
        self._resource_manager = resource_manager
        self._mic_topic = mic_topic
        self._vad_topic = vad_topic

        self._topic_worker = None
        self._executor = None
        self._tasks = dict()
        self._stopped = ThreadsafeBoolean()

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._mic_topic], self._event_bus, provides=[self._vad_topic],
                                         resource_manager=self._resource_manager, processor=self._process)
        self._topic_worker.start().wait()
        self._executor = ThreadPoolExecutor(max_workers=2)

    def stop(self):
        if not self._topic_worker:
            pass

        self._stopped.value = True
        self._topic_worker.stop()
        self._topic_worker.await_stop()
        self._executor.shutdown(wait=False)
        self._topic_worker = None
        self._executor = None

    def _process(self, event):
        payload = event.payload
        if event.payload.type == AudioSignalStarted.__name__:
            self._tasks[payload.signal_id] = self._executor.submit(self._vad_task(payload))
        if event.payload.type == AudioSignalStopped.__name__:
            if payload.signal_id not in self._tasks:
                logger.error("Received AudioStopped without running VAD: %s", event)
                return
            self._tasks[payload.signal_id].result()
            del self._tasks[payload.signal_id]

    def _vad_task(self, payload):
        audio_id, url = (payload.signal_id, payload.files[0])

        self._stopped.value = False

        def detect():
            source_offset = 0
            while not self._stopped.value:
                speech, offset, consumed, frame_size = self._listen(url, source_offset)
                speech = list(speech)

                speech_offset = source_offset + (offset * frame_size)
                vad_event = self._create_payload(speech, speech_offset, payload)
                self._event_bus.publish(self._vad_topic, Event.for_payload(vad_event))

                source_offset += consumed * frame_size

        return detect

    def _listen(self, url, offset):
        with self._audio_loader(url, offset, -1) as source:
            return self._vad.detect_vad(source, source.rate, blocking=True) + (source.frame_size,)

    def _create_payload(self, speech, speech_offset, payload):
        segment = Index.from_range(payload.signal_id, speech_offset, speech_offset + sum(len(frame) for frame in speech))
        annotation = VadAnnotation.for_activation(1.0, self._vad.__class__.__name__)

        return VadMentionEvent.create(segment, annotation)
