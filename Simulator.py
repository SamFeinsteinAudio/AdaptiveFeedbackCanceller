import os
from scipy.io import wavfile

def generate_sin(freq=440, scaling=0.2):
    return 0


if __name__ == "__main__":
    filepath = input("Provide a wav file for feedback simulation or say 'sin' for an idealized 440 Hz Sine Tone Simulation")
    if filepath.strip().lower() == "sin":
        samplerate = 48000
        signal = generate_sin(freq=440, scaling=0.2)
    elif os.path.isfile(filepath):
        samplerate, data = wavfile.read(filepath)
        # Extract signal as numpy
    else:


