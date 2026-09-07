from scipy import signal
import numpy as np


class BandStopFilter:
    """Causal, stateful Butterworth band-stop filter for streaming audio."""

    def __init__(self, low_freq=250, high_freq=8000, order=4, samplerate=44100):
        self.samplerate = samplerate
        self.second_order_sections = None
        self._filter_state = None
        self.calculate_filter_params(low_freq, high_freq, order)

    def calculate_filter_params(self, low_freq, high_freq, order):
        nyquist = self.samplerate / 2
        if not 0 < low_freq < high_freq < nyquist:
            raise ValueError(
                "Band-stop frequencies must satisfy "
                f"0 < low_freq < high_freq < {nyquist}."
            )

        self.second_order_sections = signal.butter(
            order,
            [low_freq / nyquist, high_freq / nyquist],
            btype="bandstop",
            output="sos",
        )
        self._filter_state = np.zeros((len(self.second_order_sections), 2))
        return self.second_order_sections

    def process_sample(self, sample):
        filtered, self._filter_state = signal.sosfilt(
            self.second_order_sections,
            np.asarray([sample], dtype=float),
            zi=self._filter_state,
        )
        return float(filtered[0])

    def process_buffer(self, samples):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1:
            raise ValueError("samples must be a one-dimensional array")
        filtered, self._filter_state = signal.sosfilt(
            self.second_order_sections, samples, zi=self._filter_state
        )
        return filtered
