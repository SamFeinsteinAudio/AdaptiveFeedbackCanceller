from SourceSignals.base_source_signal import SourceSignal
import numpy as np

class SinGenerator(SourceSignal):
    def __init__(self, samplerate=44100):
        self.last_sample = 0
        self.samplerate=samplerate

    def get_next_sample(self, scaling=1.0, freq=440.0, iterate=True, **kwargs):
        sample_value = scaling * np.sin(2.0 * np.pi * freq * self.last_sample / self.samplerate)
        if iterate:
            self.last_sample += 1  #TODO: Implement a wrap-around to keep from getting too big
        return sample_value

    def get_buffer(self, num_samples, scaling=1.0, freq=440.0, iterate=True, **kwargs):
        time_array = np.linspace(0, self.last_sample, self.last_sample+num_samples, endpoint=False)
        value_array = scaling * np.sin(2.0 * np.pi * freq * time_array)
        if iterate:
            self.last_sample+=num_samples  #TODO: Implement a wrap-around to keep from getting too big
        return value_array

