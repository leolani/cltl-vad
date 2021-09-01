from queue import Queue
from typing import Iterable, Any


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