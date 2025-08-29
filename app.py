import streamlit as st
import soundfile as sf
import tempfile
import whisper
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np

# Ensure reproducibility in langdetect
DetectorFactory.seed = 0

# Load Whisper model (tiny = fastest, change to "base" or "small" for better accuracy)
@st.cache_resource
def load_model():
    return whisper.load_model("tiny")

model = load_model()

st.title("🎤 MyHeritageAI – Multilingual Story Collector")

# ---------- File Upload ----------
st.header("📂 Upload Audio")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

def process_audio(path):
    """Transcribe, detect language, and translate."""
    with st.spinner("Transcribing..."):
        result = model.transcribe(path, fp16=False)
        transcript = result["text"]

    st.subheader("📝 Transcript")
    st.write(transcript)

    # Detect language
    try:
        lang = detect(transcript)
        st.success(f"Detected Language: {lang}")
    except Exception as e:
        st.error(f"Language detection failed: {e}")
        lang = None

    # Translate (if not English)
    if lang and lang != "en":
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(transcript)
            st.subheader("🌍 Translation (English)")
            st.write(translated)
        except Exception as e:
            st.error(f"Translation failed: {e}")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        data, samplerate = sf.read(uploaded_file)
        sf.write(tmp.name, data, samplerate)
        tmp_path = tmp.name

    st.audio(uploaded_file, format="audio/wav")
    process_audio(tmp_path)

# ---------- Live Recording ----------
st.header("🎙️ Record from Mic")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray().astype(np.float32).flatten()
        self.frames.append(audio)
        return frame

ctx = webrtc_streamer(
    key="speech-capture",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
)

if ctx and ctx.state.playing:
    if st.button("🔴 Stop & Transcribe"):
        if ctx.audio_processor:
            # Convert frames to WAV
           if ctx.audio_processor.frames:
    audio = np.concatenate(ctx.audio_processor.frames)
else:
    st.warning("No audio captured yet. Please record again.")
    st.stop()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio, 16000)  # write as 16kHz WAV
                tmp_path = tmp.name
            process_audio(tmp_path)
