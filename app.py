import streamlit as st
import tempfile
from pathlib import Path

from src.auralis.audio.features import extract_features
from src.auralis.audio.similarity import compare_features
from src.auralis.emotion.emotion import map_emotion


st.set_page_config(page_title="Auralis", layout="centered")

st.title("🎵 Auralis")
st.subheader("Frequency- and Emotion-Aware Music Analysis System")

st.markdown("---")

st.write("Upload one or two audio files to analyze emotion and similarity.")

file1 = st.file_uploader("Upload Audio File 1", type=["wav", "mp3"])
file2 = st.file_uploader("Upload Audio File 2 (optional)", type=["wav", "mp3"])

def save_temp_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


if file1 is not None:
    path1 = save_temp_file(file1)
    f1 = extract_features(path1)
    e1 = map_emotion(f1.meta)

    st.markdown("### 🎧 File 1 Emotion Prediction")
    st.write("Predicted Emotion:", e1.emotion)
    st.write("Scores:", e1.scores)

    if file2 is not None:
        path2 = save_temp_file(file2)
        f2 = extract_features(path2)
        e2 = map_emotion(f2.meta)

        st.markdown("### 🎧 File 2 Emotion Prediction")
        st.write("Predicted Emotion:", e2.emotion)
        st.write("Scores:", e2.scores)

        sim = compare_features(f1, f2)

        st.markdown("### 🔍 Similarity Analysis")
        st.write("Cosine Similarity:", round(sim["similarity"], 4))
