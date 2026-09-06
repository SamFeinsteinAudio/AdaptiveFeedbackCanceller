import numpy as np
from scipy import signal
import pyloudnorm
from mosqito.utils import load
from mosqito.sq_metrics import loudness_zwst, loudness_zwtv


def calc_rms(signal_arr):
    return np.sqrt(np.mean(signal_arr**2))

def calc_db(scalar):
    if scalar == 0:
        return 0
    return 20*np.log10(scalar)

def calc_dbrms(signal_arr):
    return calc_db(calc_rms(signal_arr))

def calc_scalar(db):
    return 10 ** (db/20)

def estimate_lag_and_gain(input_signal, ref):
    # returns in samples and scalar (non db)
    correlation_matrix = signal.correlate(input_signal, ref)
    max_correlation = np.argmax(np.abs(correlation_matrix))
    potential_lags = signal.correlation_lags(len(input_signal), len(ref))
    valid_indices = np.where(potential_lags > 0)[0]
    estimated_lag = potential_lags[np.argmax(correlation_matrix[valid_indices])]
    ref_power = np.sum(ref ** 2)
    if ref_power < 1e-6:
        estimated_gain = 0
    else:
        estimated_gain = correlation_matrix[max_correlation] / ref_power
    return estimated_lag, estimated_gain

def calc_lufs(input_signal, samplerate=44100):
    meter = pyloudnorm.Meter(samplerate)
    return meter.integrated_loudness(input_signal)

def estimate_loudness(input_signal, calibration=2, samplerate=44100):
    output = loudness_zwtv(input_signal*calibration, samplerate)
    return output['values'], output['time']