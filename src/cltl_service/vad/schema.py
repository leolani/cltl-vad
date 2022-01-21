import uuid

from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass

from emissor.representation.scenario import Mention, Annotation


# TODO move to cltl.combot
@dataclass
class MentionEvent:
    type: str
    mention: Mention


@dataclass
class VadMentionEvent(MentionEvent):
    @classmethod
    def create(cls, segment, annotation):
        return cls(cls.__name__, Mention(str(uuid.uuid4()), [segment], [annotation]))


@dataclass
class VadAnnotation(Annotation[float]):
    @classmethod
    def for_activation(cls, activation: float, source: str):
        return cls(cls.__name__, activation, source, timestamp_now())