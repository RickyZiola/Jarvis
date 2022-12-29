
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.io.wavfile
from scipy.signal import lfilter
import os
def denoise_waveform(waveform, cutoff_frequency, sr):
    # Convert the waveform array to a 1D numpy array
    waveform = np.array(waveform).flatten()
    
    # Normalize the cutoff frequency to the Nyquist frequency
    normalized_cutoff_frequency = cutoff_frequency / (sr / 2)
    
    # Define the filter coefficients for a low-pass filter
    b, a = scipy.signal.butter(4, normalized_cutoff_frequency, 'lowpass', analog=False)
    
    # Apply the low-pass filter to the waveform
    filtered_waveform = lfilter(b, a, waveform)
    
    return filtered_waveform

# Define the TextToSpeech class
class TextToSpeech(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x, (hidden, cell) = self.lstm(x)
        x = self.fc(x)
        return x
def split_data(text_data, audio_data):
    split_text_data  = []
    split_audio_data = []
    for line in text_data:
        l = line.split("+")
        split_text_data.append(l[0])
        split_audio_data.append(audio_data[int(l[1]):int(l[2])])
    return zip(split_text_data, split_audio_data)
def speak(model, text, device):
    # Preprocess the text input
    while len(text) < input_dim:
        text += " "
    text = preprocess_transcript(text)
    # Convert the text to a tensor
    text = torch.Tensor(text).to(device=device, dtype=torch.float32).unsqueeze(0)
    # Run the model on the text
    audio = model(text).cpu().detach().squeeze(0)
    # Convert the audio tensor to a waveform
    waveform = audio.detach().numpy() * 5
    #print(len(waveform))
    #plt.plot(waveform, color='red')
    sd.play(waveform, sr)
    #plt.pause(1)
    sd.wait()
# Define the train function
def train(model, audio_data, text_data, batch_size, epochs):
    if(1):
        clear()
        #sd.play(audio, sr)
        for i, (text, audio) in enumerate(zip(text_data, audio_data)):
            for c in text:
                print(chr(c),end="")
            while len(text) < input_dim:
                text.append(ord(' '))
            print("\nText array padded")
            audio = np.pad(audio, ((0, output_dim - len(audio)), (0, 0)), 'constant', constant_values=0)
            print("Audio array padded")
            text_data[i]  =  text
            audio_data[i] = audio
        audio_data = np.array(audio_data)
        print("Initialising tensors...")
        # Convert audio data and text data to tensors
        audio_tensor = torch.empty(audio_data.shape, dtype=torch.float32, device=device)

        # Copy the data from the NumPy array to the tensor
        audio_tensor.copy_(torch.Tensor(audio_data))
        audio_data = audio_tensor
        print("Audio tensor started")
        #print(audio_data)
        #plt.plot(audio_data.numpy())
        #plt.pause(.5)
        text = torch.Tensor(text_data).to(device, dtype=torch.float32)
        print("Text tensor started")
        # Create a dataset from the audio and text data
        dataset = list(zip(text, audio_data))
        #print(dataset)
        # Create a data loader from the dataset
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Data loader defined")
        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters())
        print("Starting training...")
        # Train the model
        for epoch in range(epochs):
            for text_d, audio_d in data_loader:
                optimizer.zero_grad()
                output = model(text_d)[0]
                loss = criterion(output, audio_d[0,:,0])
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch} out of {epochs}. Loss: {loss.item()}", end="        \r")
            if epoch % 500 == 0 and epoch != 0:
                speak(model, "rat", device)
                torch.save(model, "jarvis.pt")

def preprocess_audio(audio_path):
    # Load the audio file
    sample_rate, audio = scipy.io.wavfile.read(audio_path)
    # Convert the audio data to a numpy array
    audio = np.array(audio, dtype=np.float32)
    # Normalize the audio data
    audio /= np.max(np.abs(audio))
    return audio, sample_rate
def clear():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')

# Preprocess text data by encoding characters as integers
def preprocess_transcript(transcript):
    # Encode characters as integers
    text_data = [ord(c) for c in transcript]
    return text_data

# Load audio data and transcript
audio_data, sr = preprocess_audio("sample.wav")
text_data  = open("sample.txt", "r").readlines()
print(text_data)
text       = [n[0] for n in split_data(text_data, audio_data)]
audio_data = [n[1] for n in split_data(text_data, audio_data)]
try:
    sd.play(audio_data[text.index(input("Enter a word >>> "))])
    sd.wait()
except:pass
print(text)
text_data  = [preprocess_transcript(t) for t in text]
#print(audio_data)
#print(f"Audio length is {len(audio_data)/sr}s")
#print(f"Text length is  {len(text_data)}")
def max_len(lst):
    maxList = max(lst, key = lambda i: len(i))
    maxLength = len(maxList)
     
    return maxList, maxLength
# Define the model
input_dim = 50
hidden_dim = 128
output_dim = 160000

# Move the model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = TextToSpeech(input_dim, hidden_dim, output_dim).to(device=device)

# Train the model
batch_size = 32
epochs = 100000
if __name__ == "__main__":
    train(model, audio_data, text_data, batch_size, epochs)

    # Save the model to a file
    torch.save(model, "jarvis.pt")