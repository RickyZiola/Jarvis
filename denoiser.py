import numpy as np
import scipy.io.wavfile
import scipy.signal
import sounddevice as sd

def remove_frequency(audio, sampling_rate, frequency):
    # Calculate the number of samples in the audio signal
    num_samples = audio.shape[0]
    
    # Create a band-stop filter using the Butterworth design
    b, a = scipy.signal.butter(4, [frequency - 1, frequency + 1], 'stop', analog=False, fs=sampling_rate)
    
    # Apply the band-stop filter to the audio signal
    audio_filtered = scipy.signal.filtfilt(b, a, audio, padlen = 1)
    
    # Return the filtered audio signal
    return audio_filtered


def main():
    # Prompt the user for the audio filename
    filename = input("Enter the audio filename: ")
    
    # Load the audio file
    sampling_rate, audio = scipy.io.wavfile.read(filename)
    
    # Calculate the average frequency from the start and end of the file
    start_frequency = np.mean(np.abs(np.fft.rfft(audio[int(sampling_rate / 10):int(sampling_rate / 5) * 3])))
    end_frequency = np.mean(np.abs(np.fft.rfft(audio[int(-sampling_rate / 5) * 3:int(-sampling_rate / 10)])))
    frequency = (start_frequency + end_frequency) / 2
    frequency_hz = frequency * sampling_rate / audio.shape[0]
    print(frequency_hz)
    # Remove the frequency from the audio
    audio_filtered = remove_frequency(audio, sampling_rate, frequency)
    
    # Play the filtered audio
    sd.play(audio_filtered, sampling_rate)
    sd.wait()
main()