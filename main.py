import streamlit as st
import whisper
import datetime
import subprocess
import torch
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import wave
import contextlib
import os
import warnings
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
import tempfile

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize speaker embedding model globally
@st.cache_resource
def load_speaker_embedding_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    try:
        subprocess.run(['ffmpeg', '-i', input_path, '-acodec', 'pcm_s16le', '-ar', '16000', output_path, '-y'],
                       capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        st.error(f"Error converting audio: {str(e)}")
        return False

def process_audio(audio_file, num_speakers, language, model_size):
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file
            input_path = os.path.join(temp_dir, "input_audio")
            with open(input_path, "wb") as f:
                f.write(audio_file.getbuffer())

            # Convert to WAV
            wav_path = os.path.join(temp_dir, "audio.wav")
            if not convert_audio_to_wav(input_path, wav_path):
                return None, "Failed to convert audio file to WAV format"

            # Verify the WAV file exists
            if not os.path.exists(wav_path):
                return None, "WAV file not created successfully"

            # Load the Whisper model
            model_name = model_size
            if language == 'English' and model_size != 'large':
                model_name += '.en'

            model = whisper.load_model(model_size)
            result = model.transcribe(wav_path)
            segments = result["segments"]

            # Get audio duration
            with contextlib.closing(wave.open(wav_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)

            # Get speaker embeddings
            embedding_model = load_speaker_embedding_model()

            def segment_embedding(segment):
                start = segment["start"]
                end = min(duration, segment["end"])

                # Load audio segment
                waveform, sample_rate = torchaudio.load(
                    wav_path,
                    frame_offset=int(start * rate),
                    num_frames=int((end - start) * rate)
                )

                # Get embedding
                with torch.no_grad():
                    embedding = embedding_model.encode_batch(waveform)
                    return embedding.squeeze().cpu().numpy()

            # Get embeddings for each segment
            embeddings = np.zeros(shape=(len(segments), 192))
            for i, segment in enumerate(segments):
                embeddings[i] = segment_embedding(segment)

            embeddings = np.nan_to_num(embeddings)

            # Cluster embeddings
            clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
            labels = clustering.labels_

            # Add speaker labels to segments
            for i in range(len(segments)):
                segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

            return segments, None

    except Exception as e:
        return None, f"Error processing audio: {str(e)}"

def time(secs):
    return datetime.timedelta(seconds=round(secs))

# Streamlit UI
st.set_page_config(
    page_title="Speaker Diarization",
    page_icon="🎙️",
    layout="wide"
)

st.title("🎙️ Speaker Diarization and Transcription")

# Add CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Check for ffmpeg
try:
    subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
except (subprocess.CalledProcessError, FileNotFoundError):
    st.error("❌ FFmpeg is not installed. Please install FFmpeg to use this application.")
    st.stop()

# File upload
st.markdown("### Upload Audio File")
st.info("🎵 Supported formats: MP3, WAV (Will be converted to 16kHz WAV for processing)")
audio_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav'])

# Play audio button to verify the uploaded file
if audio_file is not None:
    st.audio(audio_file, format='audio/wav')

# Parameters
with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        num_speakers = st.number_input("Number of speakers",
                                       min_value=2,
                                       value=2,
                                       help="Specify how many different speakers are in the audio")
    with col2:
        language = st.selectbox("Language",
                                ['English', 'any'],
                                help="Select 'English' for better performance on English audio")
    with col3:
        model_size = st.selectbox("Model size",
                                  ['tiny', 'base', 'small', 'medium', 'large'],
                                  help="Larger models are more accurate but slower")

if audio_file is not None:
    if st.button("🎯 Process Audio", key="process_button"):
        with st.spinner("🔄 Processing audio file... This may take a few minutes."):
            try:
                segments, error = process_audio(audio_file, num_speakers, language, model_size)

                if error:
                    st.error(f"❌ {error}")
                elif segments:
                    st.success("✅ Audio processed successfully!")

                    st.subheader("📝 Transcript")
                    transcript = ""
                    for i, segment in enumerate(segments):
                        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                            transcript += f"\n{segment['speaker']} {time(segment['start'])}\n"
                        transcript += segment["text"] + ' '

                    st.text_area("Full transcript", transcript, height=400)

                    st.download_button(
                        label="📥 Download Transcript",
                        data=transcript,
                        file_name="transcript.txt",
                        mime="text/plain",
                        key="download_button"
                    )

            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")

# Help section
with st.expander("ℹ️ Help & Tips"):
    st.markdown("""
    ### Prerequisites:
    - FFmpeg must be installed on your system
    - Sufficient disk space for temporary files

    ### Tips for best results:
    1. **Audio Quality**: Better quality audio will yield more accurate results
    2. **Model Selection**:
        - Use larger models for more accuracy (but slower processing)
        - Use smaller models for faster results
    3. **Language Selection**:
        - Choose 'English' if your audio is in English
        - Choose 'any' for other languages
    4. **Number of Speakers**:
        - Set this to the exact number of speakers in your audio
        - Incorrect speaker counts may lead to less accurate diarization

    ### Troubleshooting:
    - If processing fails, try with a smaller model size
    - Ensure your audio file isn't corrupted
    - Check that FFmpeg is properly installed
    - Make sure you have enough free disk space
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Whisper, and SpeechBrain")
