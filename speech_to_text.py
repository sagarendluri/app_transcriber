import whisper
import time
import os

model = whisper.load_model("base")

result = model.transcribe("temp_audio.wav")
print("📜 Transcription:", result["text"])
print("-" * 50)