import numpy as np
from Filters.bandstopfilter import BandStopFilter
from SourceSignals.Noise.noise_generator import NoiseGenerator
from Measurements.audio_calculations import calc_dbrms, calc_scalar, estimate_lag_and_gain

eps = 1e-4

class LMSFeedbackCanceller:
    def __init__(self, filter_order=2048, adaptation_speed=0.15, samplerate=44100,  watermark_db_mask=30):
        self.filter_order = filter_order
        self.weights = np.zeros(filter_order)
        self.input_buffer = np.zeros(filter_order)
        self.watermark_buffer = np.zeros(filter_order)
        self.adaptation_speed = adaptation_speed
        self.watermark_generator = NoiseGenerator()
        self.watermark_filter = BandStopFilter(samplerate=samplerate)
        self.watermark_db_mask = watermark_db_mask  # level difference (pre-filtering) between signal and watermark
        # Filter between 8k and 250 to target hardest-to-hear frequencies per ISO-226
        self.watermark_filter.calculate_filter_params(250, 8000, 4)

    def process_sample(self, sample):
        self.input_buffer = np.roll(self.input_buffer, -1)
        self.input_buffer[-1] = sample

        signal_level_db = calc_dbrms(self.input_buffer)
        watermark_level_db = signal_level_db - self.watermark_db_mask
        watermark_scalar = calc_scalar(watermark_level_db)
        watermark_sample = self.watermark_generator.get_next_sample(scaling=watermark_scalar)
        filtered_watermark_sample = self.watermark_filter.process_sample(watermark_sample)

        self.watermark_buffer = np.roll(self.watermark_buffer, -1)
        self.watermark_buffer[-1] = filtered_watermark_sample
        feedback_estimation = np.dot(self.watermark_buffer, self.weights)
        error = sample - feedback_estimation

        watermark_magnitude = np.dot(self.watermark_buffer, self.watermark_buffer) + eps
        self.weights = self.weights + (self.adaptation_speed * error * self.watermark_buffer) / watermark_magnitude
        return error + filtered_watermark_sample
