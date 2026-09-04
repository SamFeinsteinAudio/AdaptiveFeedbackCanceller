import numpy as np
from Measurements.audio_calculations import calc_scalar

class FbSim:
    def __init__(self, samplerate=44100, buffer_length=2048, gain_db=-3, delay_ms=15):
        self.samplerate = samplerate
        self.buffer = np.zeros(buffer_length)
        self.gain_scalar = 1
        self.delay_samples = 0
        self.set_delay_and_gain(gain_db, delay_ms)

    def set_delay_and_gain(self, gain_db, delay_ms):
        self.gain_scalar = calc_scalar(gain_db)
        self.delay_samples = self.samplerate * delay_ms / 1000.0
        if self.delay_samples > len(self.buffer):
            self.resize_buffer(self.delay_samples)
        return True

    def resize_buffer(self, samples_needed):
        new_size = 1 << samples_needed.bit_length()
        self.buffer = np.append(np.zeros(new_size - len(self.buffer)), self.buffer)

    def process_sample(self, sample):
        fb_sample = self.gain_scalar * self.buffer[-1 * self.delay_samples]
        np.roll(self.buffer, -1)
        self.buffer[-1] = sample + fb_sample
        return sample + fb_sample

    def process_buffer(self, buffer):
        bsize = len(buffer)
        if bsize+self.delay_samples > bsize:
            self.resize_buffer(bsize+self.delay_samples)
        fb_buffer = self.gain_scalar * self.buffer[-1*bsize-self.delay_samples:-1 * self.delay_samples]
        np.roll(self.buffer, -1 * bsize)
        self.buffer[-1*bsize:] = buffer
        return buffer + fb_buffer

    