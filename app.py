
# === Imports ===
import streamlit as st
import whisper
import os
from datetime import datetime
from langdetect import detect, DetectorFactory
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import numpy as np
from pydub import AudioSegment
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pickle
from google.auth.transport.requests import Request
from deep_translator import GoogleTranslator, LibreTranslator  # ✅ Dual fallback translators

# === Google Drive API scope ===
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_and_get_service():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('client_secrets.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def upload_to_drive(file_path, drive_service, folder_id):
    file_metadata = {
        'name': os.path.basename(file_path),
        'parents': [folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    return file.get('id')

# === Whisper setup ===
DetectorFactory.seed = 0
model = whisper.load_model("base")

# === Streamlit App ===
st.set_page_config(page_title="MyHeritageAI", layout="centered")
st.title("🌍 MyHeritageAI")
st.markdown("Share your **heritage stories** — in your own language, voice, or words!")

# Optional test translation log
try:
    test = GoogleTranslator(source='auto', target='en').translate("Bonjour tout le monde")
    st.write("Translation test (Google):", test)
except Exception as e:
    st.warning(f"GoogleTranslator test failed: {e}")

# Create dataset folder
if not os.path.exists("dataset"):
    os.makedirs("dataset")

# Inputs
story_text = st.text_area("📝 Enter your story (optional)", height=200)
language = st.selectbox("🌐 Select Language", ["English", "Telugu", "Hindi", "Tamil", "Other"])
name = st.text_input("👤 Your Name (optional)")

# Upload voice file
voice_file = st.file_uploader("🎤 Or upload a voice file (MP3/WAV)", type=["mp3", "wav"])
if voice_file:
    st.audio(voice_file)

# Record voice
st.markdown("🎙️ Or record your voice:")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.frames.append(frame)
        return frame

ctx = webrtc_streamer(
    key="audio",
    mode=WebRtcMode.SENDRECV,
    audio_receiver_size=256,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
)

recorded_audio_path = None
if ctx.audio_processor:
    if st.button("💾 Save Recording"):
        frames = ctx.audio_processor.frames
        if frames:
            samples = [f.to_ndarray() for f in frames]
            samples = np.concatenate(samples)
            audio = AudioSegment(
                samples.tobytes(),
                frame_rate=frames[0].sample_rate,
                sample_width=samples.dtype.itemsize,
                channels=len(frames[0].layout.channels)
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            recorded_audio_path = f"dataset/recorded_{timestamp}.wav"
            audio.export(recorded_audio_path, format="wav")
            st.audio(recorded_audio_path)
            st.success("✅ Recording saved!")

# === Submit Story ===
if st.button("📥 Submit Story"):
    transcript = ""
    detected_lang = ""
    translated_text = ""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dataset/story_{timestamp}.txt"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Name: {name}\n")
        f.write(f"Selected Language: {language}\n")
        f.write("Story Text:\n")
        f.write(story_text.strip() + "\n")

    if voice_file:
        voice_filename = f"dataset/voice_{timestamp}.{voice_file.name.split('.')[-1]}"
        with open(voice_filename, "wb") as f:
            f.write(voice_file.read())
        result = model.transcribe(voice_filename)
        transcript = result["text"]
        detected_lang = detect(transcript)

    elif recorded_audio_path:
        result = model.transcribe(recorded_audio_path)
        transcript = result["text"]
        detected_lang = detect(transcript)

    if transcript:
        st.write("DEBUG: Transcript:", transcript)
        try:
            translated_text = GoogleTranslator(source='auto', target='en').translate(transcript)
            st.success("✅ Translated using GoogleTranslator")
        except Exception as e1:
            st.warning(f"GoogleTranslator failed: {e1}")
            try:
                translated_text = LibreTranslator(source='auto', target='en').translate(transcript)
                st.success("✅ Translated using LibreTranslator")
            except Exception as e2:
                st.error(f"Translation failed with both services.\nGoogle error: {e1}\nLibre error: {e2}")
                translated_text = ""

        with open(filename, "a", encoding="utf-8") as f:
            f.write(f"\nTranscription:\n{transcript}")
            f.write(f"\nDetected Language: {detected_lang}")
            f.write(f"\nTranslated to English:\n{translated_text}")

        st.success("✅ Story submitted successfully!")

    # Upload to Drive
    drive_service = authenticate_and_get_service()
    folder_id = "1pEEq5qRRD6b_wL93EvZipa8k-4HnRbPp"
    uploaded_file_ids = []

    try:
        uploaded_file_ids.append(upload_to_drive(filename, drive_service, folder_id))
        if voice_file:
            uploaded_file_ids.append(upload_to_drive(voice_filename, drive_service, folder_id))
        elif recorded_audio_path:
            uploaded_file_ids.append(upload_to_drive(recorded_audio_path, drive_service, folder_id))
        st.success("☁️ Files uploaded to Google Drive!")
        st.markdown(f"**Drive File IDs:** {', '.join(uploaded_file_ids)}")
    except Exception as e:
        st.error(f"🚫 Upload failed: {e}")

    if transcript.strip():
        st.markdown(f"**Transcription:** {transcript}")
        st.markdown(f"**Detected Language:** {detected_lang}")
        st.markdown(f"**Translated to English:** {translated_text if translated_text else 'Translation failed or not available.'}")
