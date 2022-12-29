import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from vetrain_cuda import TextToSpeech, preprocess_transcript, input_dim, sr, denoise_waveform
model = torch.load("jarvis.pt")

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
    waveform = audio.cpu().detach().numpy()*20
    #print(len(waveform))
    plt.plot(waveform, color='red')
    sd.play(waveform, sr)
    sd.wait()
    plt.show()

speak(model, "The compression in cylinder three appears to be low.", torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

