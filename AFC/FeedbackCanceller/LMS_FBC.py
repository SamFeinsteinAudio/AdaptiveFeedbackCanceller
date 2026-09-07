import numpy as np

from Filters.automatic_gain_control import AutomaticGainControl
from Filters.bandstopfilter import BandStopFilter
from Measurements.audio_calculations import calc_scalar
from SourceSignals.Noise.noise_generator import NoiseGenerator


class LMSFeedbackCanceller:
    """Leaky NLMS acoustic feedback canceller.

    The adaptive filter models the feedback path from prior pedal output to
    microphone input. A low-level probe remains in the pedal output to reduce
    correlation between the output and the desired musical source.
    """

    def __init__(
        self,
        filter_order=2048,
        adaptation_speed=0.08,
        samplerate=44100,
        watermark_db_mask=45,
        leakage=1e-5,
        output_limit=1.0,
        agc_enabled=True,
        agc_target_peak=0.80,
        agc_release_ms=100.0,
        seed=None,
    ):
        if filter_order < 2:
            raise ValueError("filter_order must be at least two samples")
        if not 0 < adaptation_speed <= 1:
            raise ValueError("adaptation_speed must be in (0, 1]")
        if not 0 <= leakage < 1:
            raise ValueError("leakage must be in [0, 1)")
        if output_limit <= 0:
            raise ValueError("output_limit must be positive")
        if not 0 < agc_target_peak <= output_limit:
            raise ValueError("agc_target_peak must be in (0, output_limit]")

        self.filter_order = filter_order
        self.weights = np.zeros(filter_order)
        self.adaptation_speed = adaptation_speed
        self.leakage = leakage
        self.output_limit = output_limit
        self.watermark_db_mask = watermark_db_mask
        self.watermark_generator = NoiseGenerator(seed=seed)
        self.watermark_filter = BandStopFilter(samplerate=samplerate)
        self.agc = AutomaticGainControl(
            samplerate=samplerate,
            enabled=agc_enabled,
            target_peak=agc_target_peak,
            release_ms=agc_release_ms,
        )

        # A sliding contiguous history avoids an O(filter_order) shift on
        # every sample while keeping the reference vector in age order.
        self._reference_storage = np.zeros(filter_order * 2)
        self._reference_end = filter_order
        self._last_output = 0.0
        self._input_power = 0.0
        self._power_smoothing = 0.995
        self._normalization_epsilon = 1e-10

    def _append_reference(self, sample):
        if self._reference_end == len(self._reference_storage):
            self._reference_storage[: self.filter_order] = self._reference_storage[
                self.filter_order :
            ]
            self._reference_end = self.filter_order

        self._reference_storage[self._reference_end] = sample
        self._reference_end += 1
        return self._reference_storage[
            self._reference_end - self.filter_order : self._reference_end
        ]

    def process_sample(self, sample):
        """Cancel feedback predicted from prior pedal-output samples."""
        reference = self._append_reference(self._last_output)
        feedback_estimate = float(np.dot(self.weights, reference))
        residual = sample - feedback_estimate

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

        reference_power = float(np.dot(reference, reference))
        normalized_step = self.adaptation_speed / (
            reference_power + self._normalization_epsilon
        )
        self.weights *= 1.0 - self.leakage
        self.weights += normalized_step * residual * reference
        # Bound pathological updates without changing the normal linear path.
        np.clip(self.weights, -4.0, 4.0, out=self.weights)

        output = self.agc.process_sample(residual + watermark)
        output = float(np.clip(output, -self.output_limit, self.output_limit))
        self._last_output = output
        return output
