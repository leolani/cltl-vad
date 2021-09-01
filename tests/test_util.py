import unittest
from queue import Queue

import numpy as np

from cltl.vad.util import as_iterable


class TestVADUtil(unittest.TestCase):
    def test_as_iterable(self):
        queue = Queue()

        list(map(queue.put, range(10)))
        queue.put(None)

        iterable = as_iterable(queue)

        length = 0
        for i, element in enumerate(iterable):
            self.assertEqual(i, element)
            length += 1
        self.assertEqual(10, length)

    def test_as_iterable_with_numpy(self):
        queue = Queue()

        list(map(queue.put, [np.zeros((i, 1)) for i in range(10)]))
        queue.put(None)

        iterable = as_iterable(queue)

        length = 0
        for i, element in enumerate(iterable):
            self.assertEqual(i, element.shape[0])
            length += 1
        self.assertEqual(10, length)
