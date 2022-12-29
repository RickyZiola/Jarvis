import numpy as np
import sounddevice as sd

def play_frequency(frequency, duration=2, sampling_rate=44100):
    # Calculate the number of samples for the given duration
    num_samples = int(duration * sampling_rate)
    
    # Create a sinusoidal waveform with the given frequency
    waveform = np.sin(2 * np.pi * frequency * np.arange(num_samples) / sampling_rate)
    
    # Play the waveform
    sd.play(waveform, sampling_rate)
    sd.wait()

# Prompt the user for the frequency to play
frequency = float(input("Enter the frequency to play (in Hz): "))

# Play the frequency for 2 seconds
play_frequency(frequency)