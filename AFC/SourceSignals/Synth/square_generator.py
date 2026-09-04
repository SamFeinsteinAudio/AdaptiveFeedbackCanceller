from SourceSignals.Synth.sin_generator import SinGenerator
import numpy as np

class SinWithHarmonicsGenerator(SinGenerator):
    def get_next_sample(self, scaling=1.0, freq=440.0, iterate=True, **kwargs):
        sample_value = super().get_next_sample(scaling=1.0, freq=freq, iterate=True)
        sample_value = scaling * ((sample_value > 0) - (sample_value < 0))
        return sample_value

    def get_buffer(self, num_samples, harmonics=10, scaling=1.0, freq=440.0, **kwargs):
        value_array = super().get_buffer(num_samples, scaling=1.0, freq=freq, iterate=True)
        value_array = np.sign(value_array)
        return value_array
