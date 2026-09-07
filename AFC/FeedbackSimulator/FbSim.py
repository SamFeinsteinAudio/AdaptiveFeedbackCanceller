import numpy as np

from Measurements.audio_calculations import calc_scalar


class FbSim:
    """Causal feedback-path simulator with optional reflected-path taps.

    ``impulse_response`` represents the room response after the configurable
    direct-path delay. Its first tap is the direct path, and later taps model
    reflections at whole-sample offsets. The configured gain applies to the
    complete response.
    """

    def __init__(
        self,
        samplerate=44100,
        buffer_length=2048,
        gain_db=-3,
        delay_ms=15,
        impulse_response=None,
    ):
        if samplerate <= 0:
            raise ValueError("samplerate must be positive")
        self.samplerate = samplerate
        self.impulse_response = self._validate_impulse_response(impulse_response)
        self.buffer = np.zeros(max(buffer_length, len(self.impulse_response) + 1))
        self.gain_scalar = 1.0
        self.delay_samples = 1
        self.set_delay_and_gain(gain_db, delay_ms)

    @staticmethod
    def _validate_impulse_response(impulse_response):
        if impulse_response is None:
            return np.asarray([1.0])
        impulse_response = np.asarray(impulse_response, dtype=float)
        if impulse_response.ndim != 1 or impulse_response.size == 0:
            raise ValueError("impulse_response must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(impulse_response)):
            raise ValueError("impulse_response must contain only finite values")
        return impulse_response

    def set_delay_and_gain(self, gain_db, delay_ms):
        if delay_ms <= 0:
            raise ValueError("delay_ms must be positive for a causal feedback path")
        self.gain_scalar = calc_scalar(gain_db)
        self.delay_samples = max(1, int(round(self.samplerate * delay_ms / 1000.0)))
        self._ensure_buffer_capacity(self.delay_samples + len(self.impulse_response))

    def _ensure_buffer_capacity(self, samples_needed):
        if samples_needed <= len(self.buffer):
            return
        new_size = 1 << (samples_needed - 1).bit_length()
        self.buffer = np.pad(self.buffer, (new_size - len(self.buffer), 0))

    def resize_buffer(self, samples_needed):
        """Backward-compatible capacity helper."""
        self._ensure_buffer_capacity(samples_needed)

    def process_sample(self, sample):
        start = len(self.buffer) - self.delay_samples - len(self.impulse_response) + 1
        end = len(self.buffer) - self.delay_samples + 1
        delayed_output = self.buffer[start:end][::-1]
        feedback = self.gain_scalar * float(np.dot(self.impulse_response, delayed_output))
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = sample
        return feedback

    def process_buffer(self, samples):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional array")
        return np.asarray([self.process_sample(sample) for sample in samples])
