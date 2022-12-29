print("Starting AI...", end="\n")

import speech_recognition as sr
from chatgpt_wrapper import ChatGPT
import pyttsx3
import matplotlib.pyplot as plt

# Set up the Chatbot API client

bot = ChatGPT()
print("done")
# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Set up the speech recognition
r = sr.Recognizer()
r.pause_threshold = 1
r.energy_threshold -= 50
print(r.energy_threshold)
mic = sr.Microphone()
while 1:
    # Start listening for speech
    with mic as source:
        print("Listening...")
        audio = r.listen(source, 4, 4)
    plt.plot([x for x in audio.frame_data][:1000])
    plt.show()
    # Recognize the speech and process it with the Chatbot API
    print("Processing speech...")
    text = r.recognize_sphinx(audio)
    print(f"You said: {text}")

    # Make the Chatbot API request
    for block in bot.ask_stream(text):

    # Get the response text
        print(block)
        print(f"{block}", end="")

            # Speak the response
        engine.say(block)
        engine.runAndWait()