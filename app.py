import streamlit as st
import tempfile
from pathlib import Path

from src.auralis.audio.features import extract_features
from src.auralis.audio.similarity import compare_features
from src.auralis.emotion.emotion import map_emotion


st.set_page_config(page_title="Auralis", layout="centered")

# ---------- UI HEADER ----------
st.title("🎵 Auralis")
st.subheader("Frequency- and Emotion-Aware Music Analysis System")
st.caption("Upload one or two audio files. Auralis extracts MFCC-based features, maps them to emotion scores, then compares similarity.")

st.markdown("---")


# ---------- HELPERS ----------
def save_temp_file(uploaded_file) -> str:
    """Save uploaded Streamlit file to a temporary path and return that path."""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def pretty_scores(scores: dict) -> dict:
    """Round scores for cleaner display."""
    return {k: round(float(v), 4) for k, v in scores.items()}


def file_summary(features_out):
    """Small, friendly summary block for what we extracted."""
    meta = features_out.meta
    duration = meta.get("duration_sec", None)
    sr = meta.get("sr", None)
    vec_dim = meta.get("vector_dim", None)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample rate (Hz)", int(sr) if sr else "—")
    c2.metric("Duration (sec)", round(float(duration), 2) if duration else "—")
    c3.metric("Vector dim", int(vec_dim) if vec_dim else "—")


# ---------- FILE UPLOAD ----------
st.markdown("### Upload")
file1 = st.file_uploader("Audio File 1", type=["wav", "mp3"])
file2 = st.file_uploader("Audio File 2 (optional)", type=["wav", "mp3"])


# ---------- RUN PIPELINE ----------
if file1 is None:
    st.info("Start by uploading **Audio File 1**.")
    st.stop()

with st.spinner("Analyzing File 1..."):
    path1 = save_temp_file(file1)
    f1 = extract_features(path1)
    e1 = map_emotion(f1.meta)

st.markdown("### 🎧 File 1 Results")
st.write(f"**Predicted emotion:** `{e1.emotion}`")
file_summary(f1)

with st.expander("See File 1 emotion scores"):
    st.json(pretty_scores(e1.scores))

with st.expander("See File 1 extracted feature metadata"):
    st.json(f1.meta)


# ---------- FILE 2 (OPTIONAL) ----------
if file2 is not None:
    with st.spinner("Analyzing File 2..."):
        path2 = save_temp_file(file2)
        f2 = extract_features(path2)
        e2 = map_emotion(f2.meta)

    st.markdown("### 🎧 File 2 Results")
    st.write(f"**Predicted emotion:** `{e2.emotion}`")
    file_summary(f2)

    with st.expander("See File 2 emotion scores"):
        st.json(pretty_scores(e2.scores))

    with st.expander("See File 2 extracted feature metadata"):
        st.json(f2.meta)

    # ---------- SIMILARITY ----------
    with st.spinner("Comparing similarity..."):
        sim = compare_features(f1, f2)
        similarity = float(sim["similarity"])

    st.markdown("### 🔍 Similarity Analysis")
    st.metric("Cosine similarity", round(similarity, 4))

    # Simple interpretation for demos
    if similarity >= 0.90:
        verdict = "Very similar (close in MFCC feature space)."
    elif similarity >= 0.75:
        verdict = "Somewhat similar (shared traits, but not identical)."
    else:
        verdict = "Less similar (different timbre/texture/energy patterns)."

    st.caption(verdict)

else:
    st.markdown("### 🔍 Similarity Analysis")
    st.info("Upload **Audio File 2** to compare similarity.")


st.markdown("---")
st.caption("Note: Emotion mapping is rule-based and interpretable—meant as a transparent baseline, not a clinical label.")
