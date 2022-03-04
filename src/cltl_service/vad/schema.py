import uuid
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, Annotation


@dataclass
class VadAnnotation(Annotation[float]):
    @classmethod
    def for_activation(cls, activation: float, source: str):
        return cls(cls.__name__, activation, source, timestamp_now())


@dataclass
class VadMentionEvent(AnnotationEvent[VadAnnotation]):
    @classmethod
    def create(cls, segment, annotation):
        return cls(cls.__name__, [Mention(str(uuid.uuid4()), [segment], [annotation])])

