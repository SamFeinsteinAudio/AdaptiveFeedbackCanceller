from scipy import signal
import numpy as np
from Filters.bandstopfilter import BandStopFilter
from SourceSignals.Noise.noise_generator import NoiseGenerator
from Measurements.audio_calculations import calc_dbrms, calc_scalar
import copy

class CrossCorrFeedbackCanceller:
    def __init__(self, buffer_size=2048, samplerate=44100, watermark_db_mask=30):
        # 2048 at 44.1k is 46ms. Will not work if >15meters from speaker to mic.
        self.filtered_input_buffer = np.zeros(buffer_size)
        self.input_buffer =  np.zeros(buffer_size)
        self.output_buffer = np.zeros(buffer_size)
        self.watermark_buffer = np.zeros(buffer_size)
        self.watermark_generator = NoiseGenerator()
        self.watermark_filter = BandStopFilter(samplerate=samplerate)
        # Filter between 8k and 250 to target hardest-to-hear frequencies per ISO-226
        self.watermark_filter.calculate_filter_params(250, 8000, 4)
        self.input_buffer_filter = copy.deepcopy(self.watermark_filter)
        self.initialization_progress = 0
        self.watermark_db_mask = watermark_db_mask  # level difference (pre-filtering) between signal and watermark

    def process_sample(self, sample):
        self.input_buffer[-1] = sample
        filtered_sample = self.input_buffer_filter.process_sample(sample)
        self.filtered_input_buffer[-1] = filtered_sample

        signal_level_db = calc_dbrms(self.input_buffer)
        watermark_level_db = signal_level_db - self.watermark_db_mask
        watermark_scalar = calc_scalar(watermark_level_db)
        watermark_sample = self.watermark_generator.get_next_sample(scaling=watermark_scalar)
        filtered_watermark_sample = self.watermark_filter.process_sample(watermark_sample)
        self.watermark_buffer[-1] = filtered_watermark_sample

        correlation_matrix = signal.correlate(self.filtered_input_buffer, self.watermark_buffer)
        max_correlation = np.argmax(np.abs(correlation_matrix))
        potential_lags = signal.correlation_lags(len(self.input_buffer), len(self.watermark_buffer))
        estimated_lag = potential_lags[np.argmax(correlation_matrix)]
        ref_power = np.sum(self.watermark_buffer ** 2)
        estimated_gain = correlation_matrix[max_correlation] / ref_power

        self.input_buffer = np.roll(self.input_buffer, -1)
        self.filtered_input_buffer = np.roll(self.filtered_input_buffer, -1)
        self.watermark_buffer = np.roll(self.watermark_buffer, -1)

        output_sample = sample + filtered_watermark_sample - estimated_gain * self.output_buffer[-1* estimated_lag]
        self.output_buffer = np.roll(self.output_buffer, -1)
        self.output_buffer[-1] = output_sample
        return output_sample

    def process_buffer(self, buffer):
        # TODO:   Similar to process_sample but we process the buffer as a whole.
        #  May need two helper functions depending on buffer size
        pass







