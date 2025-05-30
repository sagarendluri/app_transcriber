import streamlit as st
import pyaudio
import numpy as np
import whisper
import time

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

# Load Whisper model once
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Initialize PyAudio
p = pyaudio.PyAudio()

# Streamlit UI
st.title("🎙️ Real-time English Transcriber")
st.write("Click **Start** to begin live transcription. Click **Stop** to end.")

# Session State
if "is_listening" not in st.session_state:
    st.session_state.is_listening = False
if "transcriptions" not in st.session_state:
    st.session_state.transcriptions = []

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start"):
        st.session_state.is_listening = True
with col2:
    if st.button("⏹️ Stop"):
        st.session_state.is_listening = False

# Audio stream and transcription loop
if st.session_state.is_listening:
    st.info("⏺️ Listening... Speak now.")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    while st.session_state.is_listening:
        frames = []
        for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))

        audio_np = np.concatenate(frames).astype(np.float32) / 32768.0
        result = model.transcribe(audio_np)
        st.session_state.transcriptions.append(result["text"])
        st.write(f"📜 {result['text']}")
        time.sleep(0.5)

    stream.stop_stream()
    stream.close()
    st.success("🛑 Stopped Listening.")

# Output
st.subheader("📝 All Transcriptions:")
for i, text in enumerate(st.session_state.transcriptions):
    st.write(f"{i+1}. {text}")
    output_box = st.empty()
    import os
    from groq import Groq

    client = Groq(
        api_key="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",  # This is the default and can be omitted
    )

    prompt = f"Translate the following English text to Telugu:\n\n{text}"
    completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

    translation = ""
    for chunk in completion:
        content = chunk.choices[0].delta.content or ""
        translation += content
        output_box.markdown(f"**Translated Text:** {translation}")