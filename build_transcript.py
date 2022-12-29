import sounddevice as sd
import soundfile as sf
import keyboard
import numpy as np
import time
import os
import scipy.io.wavfile
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.signal import butter, filtfilt

def amplify(audio, gain, cutoff_frequency, sampling_rate):
    # Convert gain to decibels
    gain_db = 20 * np.log10(gain)
    
    # Calculate the number of samples in the audio signal
    num_samples = audio.shape[0]
    
    # Create a high-pass filter using the Butterworth design
    b, a = butter(4, cutoff_frequency, 'lowpass', analog=False, fs=sampling_rate)
    
    # Apply the high-pass filter to the audio signal
    audio_filtered = filtfilt(b, a, audio, padlen=0)
    
    # Amplify the audio signal
    audio_amplified = audio_filtered * gain
    
    # Return the amplified audio signal
    return audio_amplified

samplerate = 44100
recorded_data = np.array([[0, 0], [0, 0]])
output = "+0+0\n"
start = 0
# Define a callback function for sounddevice
def callback(indata, outdata, frames, time):
    outdata = np.empty(shape=(indata.shape[0], indata.shape[1]), dtype=np.float32)

def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')
with open("words.txt", "r") as file:
    lines = file.readlines()[start:]
    random.shuffle(lines)
    for i, word in enumerate(lines):
        output += word.replace("\n","")+"+"+str(len(recorded_data))
        clear()
        print(word)
        print(f"Word {i+1}")
        stream = sd.InputStream(samplerate=samplerate, channels=2, blocksize=1024)
        stream.start()
        # Record audio until the user stops the recording
        recording = True
        time_of_last_speaking = time.time()
        has_spoken = False
        print("Press space to begin recording...", end="\r")
        while not keyboard.is_pressed("space"):pass
        time.sleep(.4)
        print("\rRecording                        ")
        while recording:
            # Read a chunk of audio data from the stream
            data = stream.read(1024)[0]
            #data = amplify(data, gain=50.0, cutoff_frequency=1, sampling_rate=44100)
            # Concatenate the chunk of audio data with the rest of the recorded audio
            if(max(data[0]) > .004):
                recorded_data = np.concatenate((recorded_data, data))
            # Check if the user has stopped the recording
            if keyboard.is_pressed("space"):
                output += "+"+str(len(recorded_data))+"\n"
                recording = False
        # Stop the stream and close it
        stream.stop()
        stream.close()
        # Save the recorded audio to a file
        sf.write("sample.wav", recorded_data, samplerate)
        time.sleep(.25)
        with open("sample.txt", 'w') as file:
            file.write(output)