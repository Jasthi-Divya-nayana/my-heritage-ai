import os
import av
import numpy as np
import soundfile as sf
import streamlit as st
from datetime import datetime
from langdetect import detect, DetectorFactory
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

DetectorFactory.seed = 0

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

st.title("🎙️ MyHeritageAI - Story Collector")

st.markdown(
    """
    This app lets you **record stories**, transcribes them, detects language,
    and saves for cultural preservation.
    """
)

# Audio Recorder
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv_audio(self, frame):
        self.frames.append(frame.to_ndarray().flatten())
        return frame

# Start WebRTC streamer
ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
)

if ctx.state.playing:
    st.info("🎤 Recording... stop when ready.")

    if st.button("Save Recording"):
        if ctx.audio_receiver:
            frames = ctx.audio_receiver.get_frames(timeout=1)

            if len(frames) == 0:
                st.warning("No audio captured.")
            else:
                # Convert frames to numpy
                samples = np.concatenate(
                    [f.to_ndarray().flatten() for f in frames], axis=0
                ).astype(np.float32)

                # Normalize to -1.0 … 1.0
                samples = samples / np.max(np.abs(samples))

                # Save as WAV
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                recorded_audio_path = f"dataset/recorded_{timestamp}.wav"

                sf.write(recorded_audio_path, samples, frames[0].sample_rate)
                st.success(f"✅ Saved recording: {recorded_audio_path}")

# Text input option
st.subheader("✍️ Or type your story")
story_text = st.text_area("Write here:", "")

if st.button("Detect Language"):
    if story_text.strip():
        lang = detect(story_text)
        st.write(f"🌐 Detected language: **{lang}**")
    else:
        st.warning("Please enter some text first.")

