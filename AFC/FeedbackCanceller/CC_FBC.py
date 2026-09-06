from scipy import signal
import numpy as np
from Filters.bandstopfilter import BandStopFilter
from SourceSignals.Noise.noise_generator import NoiseGenerator
from Measurements.audio_calculations import calc_dbrms, calc_scalar, estimate_lag_and_gain
import copy

class CrossCorrFeedbackCanceller:
    def __init__(self, buffer_size=2048, samplerate=44100, watermark_db_mask=30):
        self.buffer_size=buffer_size
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
        self.watermark_db_mask = watermark_db_mask  # level difference (pre-filtering) between signal and watermark

    def process_sample(self, sample):
        self.input_buffer = np.roll(self.input_buffer, -1)
        self.input_buffer[-1] = sample
        filtered_sample = self.input_buffer_filter.process_sample(sample)
        self.filtered_input_buffer = np.roll(self.filtered_input_buffer, -1)
        self.filtered_input_buffer[-1] = filtered_sample

        signal_level_db = calc_dbrms(self.input_buffer)
        watermark_level_db = signal_level_db - self.watermark_db_mask
        watermark_scalar = calc_scalar(watermark_level_db)
        watermark_sample = self.watermark_generator.get_next_sample(scaling=watermark_scalar)
        filtered_watermark_sample = self.watermark_filter.process_sample(watermark_sample)

        estimated_lag, estimated_gain = estimate_lag_and_gain(self.filtered_input_buffer, self.watermark_buffer)

        if estimated_lag < 1:  # can't happen, so making this case harmless
            estimated_gain = 0

        output_sample = sample + filtered_watermark_sample - estimated_gain * self.output_buffer[-1* estimated_lag]
        self.output_buffer = np.roll(self.output_buffer, -1)
        self.output_buffer[-1] = output_sample

        self.watermark_buffer = np.roll(self.watermark_buffer, -1)
        self.watermark_buffer[-1] = filtered_watermark_sample
        return output_sample

    def process_buffer(self, buffer):
        buffer_size = len(buffer)
        if buffer_size < self.buffer_size:
            self.input_buffer = np.roll(self.input_buffer, -1*buffer_size)
            self.input_buffer[-1*buffer_size:] = buffer
            filtered_buffer = self.input_buffer_filter.process_buffer(buffer)
            self.filtered_input_buffer = np.roll(self.filtered_input_buffer, -1*buffer_size)
            self.filtered_input_buffer[-1*buffer_size:] = filtered_buffer

            signal_level_db = calc_dbrms(self.input_buffer)
            watermark_level_db = signal_level_db - self.watermark_db_mask
            watermark_scalar = calc_scalar(watermark_level_db)
            watermark_buffer = self.watermark_generator.get_buffer(buffer_size, scaling=watermark_scalar)
            filtered_watermark_buffer = self.watermark_filter.process_buffer(watermark_buffer)

            estimated_lag, estimated_gain = estimate_lag_and_gain(self.filtered_input_buffer, self.watermark_buffer)
            output_buffer = (buffer + filtered_watermark_buffer - estimated_gain *
                             self.output_buffer[-1*buffer_size-estimated_lag: -1*estimated_lag])
            self.output_buffer = np.roll(self.output_buffer, -1*buffer_size)
            self.output_buffer[-1*buffer_size:] = output_buffer

            self.watermark_buffer = np.roll(self.watermark_buffer, -1 * buffer_size)
            self.watermark_buffer[-1 * buffer_size:] = filtered_watermark_buffer

        else:
            self.input_buffer=buffer[-1*self.buffer_size:]
            filtered_buffer = self.input_buffer_filter.process_buffer(buffer)
            self.filtered_input_buffer = filtered_buffer[-1*self.buffer_size]

            signal_level_db = calc_dbrms(buffer)
            watermark_level_db = signal_level_db - self.watermark_db_mask
            watermark_scalar = calc_scalar(watermark_level_db)
            watermark_buffer = self.watermark_generator.get_buffer(buffer_size, scaling=watermark_scalar)
            filtered_watermark_buffer = self.watermark_filter.process_buffer(watermark_buffer)
            self.watermark_buffer = filtered_watermark_buffer[-1*self.buffer_size]

            estimated_lag, estimated_gain = estimate_lag_and_gain(filtered_buffer, self.watermark_buffer)
            output_buffer = buffer + filtered_watermark_buffer - estimated_gain * self.output_buffer[:-1*estimated_lag]

            # if more buffers of this size are incoming, we need adequate history to cancel feedback
            # input buffers do not need to be re-sized, as the incoming buffer is already big enough.
            self.watermark_buffer = filtered_watermark_buffer
            self.output_buffer = output_buffer

        return output_buffer








