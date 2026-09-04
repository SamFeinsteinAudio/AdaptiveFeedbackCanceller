from scipy import signal
import numpy as np

class BandStopFilter():
    def __init__(self, low_freq=250, high_freq=8000, order=4, samplerate=44100, buffer_length=64):
        self.samplerate = samplerate
        self.second_order_secions = self.calculate_filter_params(low_freq=low_freq, high_freq=high_freq, order=order)
        self.input_buffer = np.zeros(buffer_length)
        self.buffer_length = buffer_length

    def calculate_filter_params(self, low_freq, high_freq, order):
        nyq = self.samplerate / 2
        low, high =  low_freq / nyq, high_freq / nyq
        sos = signal.butter(order, [low, high], btype='bandstop', output='sos')
        return sos

    def process_sample(self, sample):
        self.input_buffer[-1] = sample
        filtered_buffer = signal.sosfiltfilt(self.second_order_secions, self.input_buffer)
        self.input_buffer = np.roll(self.input_buffer, -1)
        return filtered_buffer[-1]

    def process_buffer(self, buffer):
        if len(buffer) <= self.buffer_length:
            self.input_buffer[-1*len(buffer):] = buffer
        else:
            self.input_buffer[-1*self.buffer_length:] = buffer[:self.buffer_length]
        self.input_buffer = np.roll(self.input_buffer, -1)
        filtered_buffer = signal.sosfiltfilt(self.second_order_secions, buffer)
        return filtered_buffer




