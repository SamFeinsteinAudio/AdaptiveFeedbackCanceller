from SourceSignals.Synth.sin_generator import SinGenerator
import numpy as np

class SinWithHarmonicsGenerator(SinGenerator):

    def get_next_sample(self, harmonics=10, scaling=1.0, freq=440.0, **kwargs):
        sample_value = 0
        for i in range(harmonics):
            harm = i+1
            h_scale = scaling/(harm + harm**2)
            h_freq = harm*freq
            sample_value += super().get_next_sample(scaling=h_scale, freq=h_freq, iterate=False)
        self.last_sample += 1  #TODO: Implement a wrap-around to keep from getting too big
        return sample_value

    def get_buffer(self, num_samples, harmonics=10, scaling=1.0, freq=440.0, **kwargs):
        value_array = np.zeros(num_samples)
        for i in range(harmonics):
            harm = i + 1
            h_scale = scaling / (harm + harm ** 2)
            h_freq = harm * freq
            value_array += super().get_buffer(num_samples, scaling=h_scale, freq=h_freq, iterate=False)
        self.last_sample += num_samples  #TODO: Implement a wrap-around to keep from getting too big
        return value_array



