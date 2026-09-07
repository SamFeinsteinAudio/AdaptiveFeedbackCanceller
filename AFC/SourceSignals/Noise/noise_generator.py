import random

import numpy as np

from SourceSignals.base_source_signal import SourceSignal


class NoiseGenerator(SourceSignal):
    """Deterministic-per-instance white-noise source when given a seed."""

    def __init__(self, seed=None, samplerate=None):
        # samplerate is accepted so this source has the same constructor shape
        # as the synthetic source generators used by the simulator.
        self._sample_rng = random.Random(seed)
        self._buffer_rng = np.random.default_rng(seed)

    def get_next_sample(self, scaling=1.0, **kwargs):
        return self._sample_rng.uniform(-scaling, scaling)

    def get_buffer(self, num_samples: int, scaling=1.0):
        return self._buffer_rng.normal(0, scaling, num_samples)
