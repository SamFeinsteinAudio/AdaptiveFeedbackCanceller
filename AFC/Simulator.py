import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

def generate_sin(freq=440, scaling=1.0, length=10, sr=44100):
    time_array = np.linspace(0, length, int(length * sr), endpoint=False)
    sin_data = scaling * np.sin(2 * np.pi * freq * time_array)
    return sr, sin_data

def generate_square(freq=440, scaling=1.0, length=10, sr=44100):
    sr, sin_data = generate_sin(freq=freq,length=length, sr=sr)
    square_data = scaling * np.sign(sin_data)
    return sr, square_data

def generate_rich(harmonics=100, fundamental=130, length=10, sr=44100):
    sr, rich_data = generate_sin(freq=fundamental, scaling=0.5, length=length, sr=sr)
    for i in np.linspace(2, harmonics, harmonics):
        _, harmonic_data = generate_sin(freq=harmonics * fundamental, scaling=1 / (i ** 2 + i), length=length, sr=sr)
        rich_data = np.add(rich_data, harmonic_data)
        return sr, rich_data

def generate_noise(num_samples, seed=None):
    if seed:
        np.random.seed(seed)
    return np.random.normal(0,1, num_samples)

def calc_dbrms(signal_arr):
    return 20 * np.log10(np.sqrt(np.mean(signal_arr**2)))

def calc_scalar(db):
    return 10 ** (db/20)


def generate_watermark(input_signal, band_reject_sos, snr=30, seed=18):
    # 4th order filters, 24db/8v slope should sound relatively natural while still steep
    base_noise = generate_noise(input_signal.size, seed=seed)
    filt_noise = signal.sosfiltfilt(band_reject_sos, base_noise)
    og_signal_level = calc_dbrms(input_signal)
    watermark_db = og_signal_level - snr
    wmark = calc_scalar(watermark_db) * filt_noise
    return wmark

def idealized_room_simulator(og_signal, spk_out, samplerate, feedback_delay_ms=50.0, feedback_gain_db=-6):
    feedback_delay_samples = round(samplerate*feedback_delay_ms/1000)
    feedback_gain_ratio = calc_scalar(feedback_gain_db)
    mic_in = np.zeros_like(og_signal)
    for i in range(len(mic_in)):
        if i < feedback_delay_samples:
            mic_in[i] = og_signal[i]
        else:
            mic_in[i] = og_signal[i] + (spk_out[i-feedback_delay_samples])*feedback_gain_ratio
    return mic_in

def estimate_gain_and_delay(ref_signal, in_signal, band_reject_sos):
    filtered_input_signal = signal.sosfiltfilt(band_reject_sos, in_signal)
    correlation_matrix = signal.correlate(filtered_input_signal, ref_signal)
    max_correlation = np.argmax(np.abs(correlation_matrix))
    potential_lags = signal.correlation_lags(len(in_signal), len(ref_signal))
    estimated_lag = potential_lags[np.argmax(correlation_matrix)]
    ref_power = np.sum(ref_signal**2)
    estimated_gain = correlation_matrix[max_correlation] / ref_power
    return estimated_gain, estimated_lag


if __name__ == "__main__":
    filepath = input("Provide a wav file for simulation, say 'sin' for an idealized Tone Simulation, say "
                     "'rich' for an idealized harmonically rich signal, or say 'square' for a square wav \n").strip()
    print(filepath)
    if filepath.lower() == "sin":
        samplerate, data = generate_sin()
    elif filepath.lower() == "rich":
        samplerate, data = generate_rich()
    elif filepath.lower() == "square":
        samplerate, data = generate_square()
    elif os.path.isfile(filepath):
        samplerate, data = wavfile.read(filepath)
    else:
        raise ValueError(f"Provided input {filepath} was not 'sin', nor was it a valid filepath")

    nyq = 0.5*samplerate
    low, high = 250/nyq, 8000/nyq
    # Filter between 8k and 250 to target hardest-to-hear frequencies per ISO-226
    sos = signal.butter(4, [low, high], btype='bandstop', output='sos')
    # 4th order filters, 24db/8v slope should sound relatively natural while still steep

    adaptive_filter_order = 2048   # allows for up to 46ms of delay (about 15 meters)
    weights = np.zeros(adaptive_filter_order)
    input_memory_buffer = np.zeros(adaptive_filter_order)
    watermark_memory_buffer = np.zeros(adaptive_filter_order)
    adaptation_speed = 0.15
    eps = 1e-4

    n_samples = len(data)
    mic_input, estimated_feedback, output_signal = np.zeros(n_samples), np.zeros(n_samples), np.zeros(n_samples)

    # ROOM SIMULATION  TODO: Put in its own function
    gain_db = np.append(np.linspace(-6.0, -9.0, n_samples//2), np.full(-(n_samples//-2),-9.0))
    gain_scalar = calc_scalar(gain_db)

    delay_ms = np.append(np.linspace(20.0,30.0, n_samples//2), np.full(-(n_samples//-2),30.0))
    delay_samples = samplerate * delay_ms

    watermark = generate_watermark(data, sos, snr=30)
    pedal_output = data + watermark

    og_indicies = np.arange(n_samples)
    fb_indicies = np.arange(n_samples) - delay_samples
    gained_pedal_output = gain_scalar * pedal_output
    feedback_simulation = np.interp(fb_indicies, og_indicies, gained_pedal_output, left=0, right=0)
    true_error = np.zeros(n_samples)
    mic_signal = feedback_simulation + data

    for i in range(n_samples):
        input_memory_buffer[0] = mic_signal[i]
        watermark_memory_buffer[0] = watermark[i]
        feedback_estimation = np.dot(input_memory_buffer, weights)
        output = mic_input[i] - feedback_estimation
        output_signal[i] = output
        true_error[i] = abs(feedback_simulation[i] - feedback_estimation)


        watermark_magnitude = np.dot(watermark_memory_buffer, watermark_memory_buffer)
        weights = weights + (adaptation_speed * output * watermark_memory_buffer) / watermark_magnitude

        watermark_memory_buffer = np.roll(watermark_memory_buffer, 1)
        input_memory_buffer = np.roll(input_memory_buffer, 1)

    print(true_error.max())
    # plt.plot(true_error)
    # plt.show()

""""
   # STATIONARY SIMULATION # 
   
    gain_db, delay_ms = -12, 50.0

    mic_signal=idealized_room_simulator(data, pedal_output, samplerate,
                                        feedback_gain_db=gain_db, feedback_delay_ms=delay_ms)

    est_gain, est_delay = estimate_gain_and_delay(watermark, mic_signal, sos)
    print(f"Raw estimated gain scalar: {est_gain}")
    print(f"Raw estimated delay samples: {est_delay}")
    est_gain_db = 20 * np.log10(est_gain)
    est_delay_ms = 1000 * est_delay / samplerate

    print(f"Estimated gain was {est_gain_db}dB. Actual was {gain_db}db")
    print(f"Estimated delay was {est_delay_ms}ms. Actual was {delay_ms}ms")
"""











