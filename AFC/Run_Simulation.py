from SourceSignals.Synth.sin_with_harmonics_generator import SinWithHarmonicsGenerator
from SourceSignals.Synth.square_generator import SquareGenerator
from FeedbackSimulator.FbSim import FbSim
from FeedbackCanceller.CC_FBC import CrossCorrFeedbackCanceller
from FeedbackCanceller.LMS_FBC import LMSFeedbackCanceller
from Measurements.audio_calculations import calc_scalar, calc_dbrms, estimate_loudness
import random
import numpy as np
import matplotlib.pyplot as plt


SOURCES = {"square": SquareGenerator, "sin": SinWithHarmonicsGenerator}
FB_PEDALS = {"CC": CrossCorrFeedbackCanceller, "LMS": LMSFeedbackCanceller}

def run_simulation(source="square", fb_alg="CC", samplerate=44100, time=30, **kwargs):
    singer_obj = SOURCES.get(source)
    if singer_obj is None:
        raise ValueError(f"{source} not in possible list of sources {SOURCES.keys()}")
    singer = singer_obj()

    pedal_obj = FB_PEDALS.get(fb_alg)
    if pedal_obj is None:
        raise ValueError(f"{fb_alg} not in possible list of fb algos {FB_PEDALS.keys()}")
    pedal = pedal_obj()


    sound_system = FbSim(samplerate=samplerate)

    total_samples = time * samplerate
    total_samples +=  (3 - total_samples%3)  # to make it easier to split into 3 sections below
    #Trying stationary state, moving state, then stationary state (where feedback would otherwise become unstable)
    room_gain_db = np.append(np.append(np.full(int(total_samples/3), -3.0), np.linspace(-3.0,3.0, int(total_samples/3))), np.full(int(total_samples/3), 3.0))
    room_delay_ms = np.append(np.append(np.full(int(total_samples/3), 30.0), np.linspace(30.0, 15.0, int(total_samples/3))), np.full(int(total_samples/3), 15.0))

    singer_full = np.zeros(total_samples)
    output_full = np.zeros(total_samples)

    np.random.seed(18)
    random.seed(18)

    singer_scale, singer_freq = 1.0, 440
    feedback_sample = 0
    for sample_num in range(total_samples):
        sound_system.set_delay_and_gain(room_gain_db[sample_num], room_delay_ms[sample_num])
        if sample_num % (0.5*samplerate) == 0:
            print(f"second # {sample_num/samplerate}")# 120BPM is a typical tempo
            singer_db = random.uniform(-6.0, -18.0)  #typical dynamic expected from a performance
            singer_scale = calc_scalar(singer_db)
            singer_pitch = random.uniform(0.0,3.0) # representing 4 octaves in typical vocal ranges
            singer_freq = 100+2**singer_pitch
        singer_sample = singer.get_next_sample(scaling=singer_scale,freq=singer_freq)
        singer_full[sample_num] = singer_sample
        pedal_sample = pedal.process_sample(singer_sample+feedback_sample)
        output_full = pedal_sample
        feedback_sample = sound_system.process_sample(pedal_sample)

    error_signal = output_full - singer_full
    error_level = calc_dbrms(error_signal)
    signal_level = calc_dbrms(singer_full)

    print(f"SER: {signal_level-error_level}")

    error_loudness, error_time = estimate_loudness(error_signal)
    signal_loudness, signal_time = estimate_loudness(singer_full)

    plt.figure(figsize=(10, 5))
    plt.plot(signal_time, signal_loudness, label="Clean Signal", color="#1f77b4", lw=2)
    plt.plot(error_time, error_loudness, label="Isolated Error (Noise)", color="#d62728", lw=2 )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Perceived Loudness (Sones)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    #TODO Make Graph so we can what sections each algo struggled with by how much

if __name__ == "__main__":
    print("CC Simulation:")
    run_simulation()
    print("LMS Simulation:")
    run_simulation(fb_alg="LMS")




