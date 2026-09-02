import os
import numpy as np
from scipy.io import wavfile
from scipy import signal

def generate_sin(freq=440, scaling=1.0, length=10, sr=48000):
    time_array = np.linspace(0, length, int(length * sr), endpoint=False)
    sin_data = scaling * np.sin(2 * np.pi * freq * time_array)
    return sr, sin_data

def generate_square(freq=440, scaling=1.0, length=10, sr=48000):
    sr, sin_data = generate_sin(freq=freq,length=length, sr=sr)
    square_data = scaling * np.sign(sin_data)
    return sr, square_data

def generate_rich(harmonics=100, fundamental=130, length=10, sr=48000):
    sr, rich_data = generate_sin(freq=fundamental, scaling=0.5, length=length, sr=sr)
    for i in np.linspace(2, harmonics, harmonics):
        _, harmonic_data = generate_sin(freq=harmonics * fundamental, scaling=1 / (i ** 2 + i), length=length, sr=sr)
        rich_data = np.add(rich_data, harmonic_data)
        return sr, rich_data

def generate_noise(num_samples, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.normal(0,1, num_samples)

def calc_dbrms(signal):
    return 20 * np.log10(np.sqrt(np.mean(signal**2)))

def calc_scalar(db):
    return 10 ** (db/20)


def generate_watermark(signal, sr, filter_order=4, snr=30, seed=18):
    # 4th order filters, 24db/8v slope should sound relatively natural while still steep
    base_noise = generate_noise(signal.size, seed=seed)
    # Filter between 8k and 250 to target hardest-to-hear frequencies per ISO-226
    nyq = 0.5*sr
    low, high = 250/nyq, 8000/nyq
    sos = signal.butter(filter_order, [low, high], btype='bandstop', output='sos')
    filt_noise = signal.sosfiltfilt(sos,base_noise)
    og_signal_level = calc_dbrms(signal)
    watermark_db = og_signal_level - snr
    wmark = calc_scalar(watermark_db) * filt_noise
    return wmark

def idealized_room_simulator(og_signal, spk_out, samplerate, feedback_delay_ms=50, feedback_gain=-6):
    feedback_delay_samples = round(samplerate*feedback_delay_ms/1000)
    feedback_gain_ratio = calc_scalar(feedback_gain)
    mic_in = np.zeros_like(og_signal)
    for i in range(len(mic_in)):
        if i < feedback_delay_samples:
            mic_in[i] = og_signal[i]
        else:
            mic_in[i] = og_signal[i] + (spk_out[i-feedback_delay_samples])*feedback_gain_ratio
    return mic_in

def estimate_gain_and_delay(ref_signal, in_signal):
    correlation_matrix = signal.correlate(in_signal, ref_signal)
    potential_lags = signal.correlation_lags(len(in_signal), len(ref_signal))
    estimated_lag = potential_lags[np.argmax(correlation_matrix)]
    ref_power = np.sum(ref_signal**2)
    estimated_gain = correlation_matrix[estimated_lag] / ref_power
    return estimated_gain, estimated_lag


if __name__ == "__main__":
    filepath = input("Provide a wav file for simulation, say 'sin' for an idealized Tone Simulation, say "
                     "'rich' for an idealized harmonically rich signal, or say 'square' for a square wav").strip()
    if filepath.lower() == "sin":
        samplerate, data = generate_sin()
    if filepath.lower() == "rich":
        samplerate, data = generate_rich()
    if filepath.lower() == "square":
        samplerate, data = generate_square()
    elif os.path.isfile(filepath):
        samplerate, data = wavfile.read(filepath)
    else:
        raise ValueError("Provided input was not 'sin', nor was it a valid filepath")

    watermark = generate_watermark(data, samplerate, snr=30)
    pedal_output = data+watermark

    mic_signal=idealized_room_simulator(data, pedal_output, samplerate)

    est_gain, est_delay = estimate_gain_and_delay(watermark, mic_signal)









