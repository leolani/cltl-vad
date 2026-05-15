import logging

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton
from cltl.vad.webrtc_vad import WebRtcVAD
from cltl_service.vad.service import VadService

logger = logging.getLogger(__name__)


class VADContainer(InfraContainer):
    @property
    @singleton
    def vad_service(self) -> VadService:
        config = self.config_manager.get_config("cltl.vad.webrtc")
        vad = WebRtcVAD(
            config.get_int("activity_window"),
            config.get_float("activity_threshold"),
            config.get_int("allow_gap"),
            config.get_int("padding"),
        )
        return VadService.from_config(vad, self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start VAD")
        super().start()
        self.vad_service.start()

    def stop(self):
        logger.info("Stop VAD")
        self.vad_service.stop()
        super().stop()
