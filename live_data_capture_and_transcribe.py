import pyaudio
import numpy as np
import whisper
import time

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# Load Whisper model
model = whisper.load_model("base")

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("🎙️ Listening... Press Ctrl+C to stop.")

try:
    while True:
        print(f"⏺️ Recording {RECORD_SECONDS} seconds...")

        # Record audio frames
        frames = []
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(np.frombuffer(data, dtype=np.int16))

        # Combine frames and normalize to float32 [-1.0, 1.0]
        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0

        # Transcribe using Whisper
        print("🧠 Transcribing...")
        result = model.transcribe(audio_np)
        print("📜 Transcription:", result["text"])
        print("-" * 50)

except Exception as e:
    # Code to handle other exceptions
    print(f"An error occurred: {e}")


