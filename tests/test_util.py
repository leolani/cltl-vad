import unittest
from queue import Queue
from typing import Iterable

import matplotlib.pyplot as plt

import numpy as np

from cltl.vad.util import as_iterable


def plot_wav(audio_array: np.array, sampling_rate, window_size, marked):
    plt.plot(audio_array)
    plt.axvspan(marked[0], marked[1], facecolor='b', alpha=0.5)
    ticks = range(0, len(audio_array), sampling_rate // 4)
    plt.xticks(ticks, [(s * 1000) // sampling_rate for s in ticks])
    plt.show()

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
