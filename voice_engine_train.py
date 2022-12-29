import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import scipy.io.wavfile
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
    print(text_data)
    boundaries = [i for i in range(1, len(text_data)) if (chr(text_data[i-1]) == "?" or chr(text_data[i-1]) == "!" or chr(text_data[i-1]) == '.') and chr(text_data[i]).isupper()]
    boundaries.append(len(text_data))
    start = 0
    split_text_data = []
    split_audio_data = []
    for boundary in boundaries:
        split_text_data.append(text_data[start:boundary])
        split_audio_data.append(audio_data[start:boundary])
        start = boundary
    return zip(split_text_data, split_audio_data)
# Define the train function
def train(model, audio_data, text_data, batch_size, epochs):
    text  = [n[0] for n in split_data(text_data, audio_data)]
    audio_data = [n[1] for n in split_data(text_data, audio_data)]
    text_data = text
    for text, audio in zip(text_data, audio_data):
        sd.play(audio, sr)
        for c in text:
            print(chr(c),end="")
        while len(text) < input_dim:
            text.append(ord(' '))
        while len(audio) < output_dim:
            np.append(audio, 0)
        # Convert audio data and text data to tensors
        audio = torch.Tensor(audio)
        #print(audio_data)
        #plt.plot(audio_data.numpy())
        #plt.pause(.5)
        text = torch.Tensor(text).unsqueeze(0).to(dtype=torch.float32)
        # Create a dataset from the audio and text data
        dataset = list(zip(text, audio))
        #print(dataset)
        # Create a data loader from the dataset
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
                print()
        sd.wait()

# Preprocess audio data by extracting a spectrogram
import scipy.io.wavfile

def preprocess_audio(audio_path):
    # Load the audio file
    sample_rate, audio = scipy.io.wavfile.read(audio_path)
    # Convert the audio data to a numpy array
    audio = np.array(audio, dtype=np.float32)
    # Normalize the audio data
    #audio /= np.max(np.abs(audio))
    return audio, sample_rate


# Preprocess text data by encoding characters as integers
def preprocess_transcript(transcript):
    # Encode characters as integers
    text_data = [ord(c) for c in transcript]
    return text_data

# Load audio data and transcript
audio_data, sr = preprocess_audio("sample.wav")
text_data = preprocess_transcript(open("sample.txt", "r").readlines()[0])
print(sr)

#sd.play(audio_data, sr)
#sd.wait()
# Define the model
input_dim = len(text_data)
hidden_dim = 128
output_dim = audio_data.shape[0]
model = TextToSpeech(input_dim, hidden_dim, output_dim)

# Train the model
batch_size = 32
epochs = 10
if __name__ == "__main__":
    train(model, audio_data, text_data, batch_size, epochs)

    # Save the model to a file
    torch.save(model, "jarvis.pt")
def speak(model, text, device):
    # Preprocess the text input
    while len(text) < input_dim:
        text += " "
    text = preprocess_transcript(text)
    # Convert the text to a tensor
    text = torch.Tensor(text).to(device=device, dtype=torch.float32).unsqueeze(0)
    # Run the model on the text
    audio = model(text).squeeze(0)
    # Convert the audio tensor to a waveform
    waveform = audio.detach().numpy()
    #print(len(waveform))
    plt.plot(waveform, color='red')
    sd.play(waveform, sr)
    plt.pause(.5)
    sd.wait()




