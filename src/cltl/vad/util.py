from queue import Queue
from typing import Iterable, Any

import numpy as np
import sounddevice as sd
import soundfile


def as_iterable(queue: Queue) -> Iterable[Any]:
    """
    Utility function to convert a Queue into a thread safe iterable.

    Parameters
    ----------
    queue : Queue[Any]
        The queue to be converted. To stop iteration the Queue must be
        terminated with a None value.

    Returns
    -------
    Iterable[Any]
        An iterable with the content of the Queue.
    """
    next = queue.get()
    while next is not None:
        yield next
        next = queue.get()


def store_frames(frames, sampling_rate, save=None):
    if not len(frames):
        return

    audio = np.concatenate(frames)
    if save:
        soundfile.write(save, audio, sampling_rate)
    else:
        sd.play(audio, sampling_rate)
        sd.wait()


def to_decibel(frames, ref=np.iinfo(np.int16).max):
    # frames = np.vstack(frames)
    # rms_ratios = np.sqrt(np.mean(np.square(frames), axis=0))
    # rms_log = (10 * np.log10(ratio) for ratio in rms_ratio)
    # decibel = [int(np.clip(db, -60, 60)) for db in rms_log]
    # print("db", ref, decibel)
    # return sum(rms_log)

    raise NotImplementedError()