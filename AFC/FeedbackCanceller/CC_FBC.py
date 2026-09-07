import numpy as np
from scipy import signal

from Filters.automatic_gain_control import AutomaticGainControl
from Filters.bandstopfilter import BandStopFilter
from Measurements.audio_calculations import calc_scalar
from SourceSignals.Noise.noise_generator import NoiseGenerator


class CrossCorrFeedbackCanceller:
    """Streaming probe-based feedback canceller with periodic path updates.

    A low-level, band-stopped noise probe identifies the dominant delayed
    pedal-output component in the microphone signal. The identified gain and
    delay are then applied to the complete pedal-output history to estimate
    and remove acoustic feedback.
    """

    def __init__(
        self,
        buffer_size=4096,
        update_interval=256,
        samplerate=44100,
        watermark_db_mask=15,
        smoothing=0.0,
        output_limit=1.0,
        agc_enabled=True,
        agc_target_peak=0.80,
        agc_release_ms=100.0,
        maximum_lag_samples=None,
        cancellation_scale=0.4,
        seed=None,
    ):
        if buffer_size < 2:
            raise ValueError("buffer_size must be at least two samples")
        if not 0 < update_interval <= buffer_size:
            raise ValueError("update_interval must be between one and buffer_size")
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        if output_limit <= 0:
            raise ValueError("output_limit must be positive")
        if not 0 < agc_target_peak <= output_limit:
            raise ValueError("agc_target_peak must be in (0, output_limit]")
        if maximum_lag_samples is None:
            maximum_lag_samples = buffer_size // 2
        if not 1 <= maximum_lag_samples < buffer_size:
            raise ValueError("maximum_lag_samples must be in [1, buffer_size)")
        if not 0 < cancellation_scale <= 1:
            raise ValueError("cancellation_scale must be in (0, 1]")

        self.buffer_size = buffer_size
        self.cancellation_scale = cancellation_scale
        self.maximum_lag_samples = maximum_lag_samples
        self.update_interval = update_interval
        self.watermark_db_mask = watermark_db_mask
        self.smoothing = smoothing
        self.output_limit = output_limit
        self.watermark_generator = NoiseGenerator(seed=seed)
        self.watermark_filter = BandStopFilter(samplerate=samplerate)
        self.microphone_filter = BandStopFilter(samplerate=samplerate)
        self.agc = AutomaticGainControl(
            samplerate=samplerate,
            enabled=agc_enabled,
            target_peak=agc_target_peak,
            release_ms=agc_release_ms,
        )

        # Each buffer is circular. The index denotes the oldest sample, which
        # makes a delayed output lookup O(1) and avoids per-sample np.roll.
        self.input_history = np.zeros(buffer_size)
        self.watermark_history = np.zeros(buffer_size)
        self.output_history = np.zeros(buffer_size)
        self._write_index = 0
        self._samples_seen = 0
        self._input_power = 0.0
        self._power_smoothing = 0.995

        self.estimated_lag = 1
        self.estimated_gain = 0.0
        self.last_confidence = 0.0
        self.path_updates = 0

    def _history_in_time_order(self, history):
        return np.concatenate((history[self._write_index :], history[: self._write_index]))

    def _update_path_estimate(self):
        microphone = self._history_in_time_order(self.input_history)
        watermark = self._history_in_time_order(self.watermark_history)
        watermark_energy = float(np.dot(watermark, watermark))
        if watermark_energy <= 1e-12:
            return

        correlation = signal.correlate(microphone, watermark, method="fft")
        lags = signal.correlation_lags(len(microphone), len(watermark))
        valid_lags = lags[(lags > 0) & (lags <= self.maximum_lag_samples)]
        correlation_offset = len(watermark) - 1
        valid_correlation = correlation[correlation_offset + valid_lags]

        # Normalize each lag by only the overlapping signal energy. Without
        # this, near-buffer-length lags look artificially strong because the
        # denominator includes watermark samples that did not overlap.
        microphone_energy_prefix = np.concatenate(
            ([0.0], np.cumsum(microphone * microphone))
        )
        watermark_energy_prefix = np.concatenate(
            ([0.0], np.cumsum(watermark * watermark))
        )
        microphone_overlap_energy = microphone_energy_prefix[-1] - microphone_energy_prefix[
            valid_lags
        ]
        watermark_overlap_energy = watermark_energy_prefix[self.buffer_size - valid_lags]
        normalization = np.sqrt(
            np.maximum(microphone_overlap_energy * watermark_overlap_energy, 1e-24)
        )
        normalized_correlation = np.abs(valid_correlation) / normalization
        peak_index = int(np.argmax(normalized_correlation))
        candidate_lag = int(valid_lags[peak_index])
        overlap_watermark_energy = float(watermark_overlap_energy[peak_index])
        closed_loop_gain = float(valid_correlation[peak_index] / overlap_watermark_energy)
        # A probe circulates through the closed feedback loop. For a scalar
        # path h, its observed gain is h / (1 - h); invert that relation so
        # cancellation uses the physical single-pass path gain h.
        candidate_gain = closed_loop_gain / (1.0 + closed_loop_gain)

        self.last_confidence = float(normalized_correlation[peak_index])
        # The feedback path in this simulator is causal and positive. Reject
        # implausible correlation peaks caused by source/probe coincidence.
        if not 0.0 < closed_loop_gain or not 0.0 < candidate_gain < 0.95:
            return

        if self.path_updates == 0:
            self.estimated_lag = candidate_lag
            self.estimated_gain = candidate_gain
        else:
            self.estimated_lag = max(
                1,
                min(
                    self.buffer_size - 1,
                    int(
                        round(
                            self.smoothing * self.estimated_lag
                            + (1 - self.smoothing) * candidate_lag
                        )
                    ),
                ),
            )
            self.estimated_gain = (
                self.smoothing * self.estimated_gain
                + (1 - self.smoothing) * candidate_gain
            )
        self.path_updates += 1

    def process_sample(self, sample):
        """Process one microphone sample and return the compensated output."""
        filtered_sample = self.microphone_filter.process_sample(sample)
        if (
            self._samples_seen >= self.buffer_size
            and self._samples_seen % self.update_interval == 0
        ):
            self._update_path_estimate()

        self._input_power = self._power_smoothing * self._input_power + (
            1 - self._power_smoothing
        ) * sample**2
        watermark_level_db = max(
            20 * np.log10(max(np.sqrt(self._input_power), 1e-6))
            - self.watermark_db_mask,
            -90.0,
        )
        watermark = self.watermark_filter.process_sample(
            self.watermark_generator.get_next_sample(
                scaling=calc_scalar(watermark_level_db)
            )
        )

        delayed_output_index = (self._write_index - self.estimated_lag) % self.buffer_size
        # The conservative scale leaves margin for closed-loop estimation bias
        # and for non-ideal source/probe separation.
        feedback_estimate = (
            self.cancellation_scale
            * self.estimated_gain
            * self.output_history[delayed_output_index]
        )
        output = self.agc.process_sample(sample - feedback_estimate + watermark)
        output = float(np.clip(output, -self.output_limit, self.output_limit))

        self.input_history[self._write_index] = filtered_sample
        self.watermark_history[self._write_index] = watermark
        self.output_history[self._write_index] = output
        self._write_index = (self._write_index + 1) % self.buffer_size
        self._samples_seen += 1
        return output

    def process_buffer(self, samples):
        """Process a buffer through the same stateful path as process_sample."""
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional array")
        return np.asarray([self.process_sample(sample) for sample in samples])
