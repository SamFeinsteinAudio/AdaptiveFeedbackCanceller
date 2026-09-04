from SourceSignals.base_source_signal import SourceSignal
import numpy as np
import random

class NoiseGenerator(SourceSignal):
    def __init__(self, seed=None):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

    def get_next_sample(self, scaling=1.0, **kwargs):
        return random.uniform(-1.0 * scaling, scaling)

    def get_buffer(self, num_samples:int, scaling=1.0):
        return np.random.normal(0, scaling, num_samples)
