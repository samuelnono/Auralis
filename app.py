import streamlit as st
import tempfile
from pathlib import Path

from src.auralis.audio.features import extract_features
from src.auralis.audio.similarity import compare_features
from src.auralis.emotion.emotion import map_emotion
from src.auralis.preference.profile import UserProfile
from src.auralis.preference.feedback import record_feedback, feedback_summary
from src.auralis.preference.recommender import rank_songs, load_index

# ── Config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Auralis", layout="wide")

PROFILE_PATH = "data/processed/user_profile.json"
INDEX_PATH   = "data/processed/research_index.csv"


# ── Session state bootstrap ──────────────────────────────────────────────────
if "profile" not in st.session_state:
    st.session_state.profile = UserProfile.load_or_new(PROFILE_PATH)

if "rated_paths" not in st.session_state:
    st.session_state.rated_paths = set(
        e["track"] for e in st.session_state.profile.interaction_log
    )

if "last_features" not in st.session_state:
    st.session_state.last_features = None   # FeatureOutput of most recent upload
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None    # EmotionOutput of most recent upload


# ── Helpers ──────────────────────────────────────────────────────────────────
def save_temp_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def pretty_scores(scores: dict) -> dict:
    return {k: round(float(v), 4) for k, v in scores.items()}


def file_summary(features_out):
    meta = features_out.meta
    c1, c2, c3 = st.columns(3)
    c1.metric("Sample rate (Hz)", int(meta.get("sr", 0) or 0))
    c2.metric("Duration (sec)",   round(float(meta.get("duration_sec", 0) or 0), 2))
    c3.metric("Vector dim",       int(meta.get("vector_dim", 0) or 0))


def emotion_bar(scores: dict):
    """Render a small horizontal bar chart for emotion scores."""
    import pandas as pd
    df = pd.DataFrame(
        {"Emotion": list(scores.keys()), "Score": [round(v, 3) for v in scores.values()]}
    ).set_index("Emotion")
    st.bar_chart(df)


def _apply_feedback(feat, emo, label: str):
    """Handle a like or dislike from any part of the UI."""
    st.session_state.profile = record_feedback(
        profile=st.session_state.profile,
        vector=feat.vector,
        emotion_scores=emo.scores,
        emotion_label=emo.emotion,
        track_path=feat.meta["path"],
        label=label,
        profile_save_path=PROFILE_PATH,
    )
    st.session_state.rated_paths.add(feat.meta["path"])


# ── Navigation ───────────────────────────────────────────────────────────────
st.sidebar.title("🎵 Auralis")
st.sidebar.caption("Frequency- and Emotion-Aware Music Recommendation")
st.sidebar.markdown("---")
tab_choice = st.sidebar.radio(
    "Navigate",
    ["🎧 Analyze", "⭐ Recommendations", "👤 My Profile"],
)

# Show compact profile snapshot in sidebar
profile = st.session_state.profile
if profile.has_signal():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Your taste so far**")
    st.sidebar.caption(f"Dominant emotion: `{profile.dominant_emotion()}`")
    st.sidebar.caption(f"👍 {profile.total_likes}  👎 {profile.total_dislikes}")


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ════════════════════════════════════════════════════════════════════════════
if tab_choice == "🎧 Analyze":
    st.title("🎧 Analyze Audio")
    st.caption(
        "Upload one or two audio files. Auralis extracts MFCC features, "
        "maps them to emotion scores, and compares acoustic similarity."
    )
    st.markdown("---")

    file1 = st.file_uploader("Audio File 1", type=["wav", "mp3"])
    file2 = st.file_uploader("Audio File 2 (optional — for similarity)", type=["wav", "mp3"])

    if file1 is None:
        st.info("Upload **Audio File 1** to get started.")
        st.stop()

    with st.spinner("Analyzing File 1…"):
        path1 = save_temp_file(file1)
        f1 = extract_features(path1)
        e1 = map_emotion(f1.meta)
        st.session_state.last_features = f1
        st.session_state.last_emotion  = e1

    col_res, col_act = st.columns([3, 1])
    with col_res:
        st.markdown("### File 1 Results")
        st.write(f"**Predicted emotion:** `{e1.emotion}`")
        file_summary(f1)
        with st.expander("Emotion scores"):
            emotion_bar(e1.scores)
        with st.expander("Raw feature metadata"):
            st.json(f1.meta)

    with col_act:
        st.markdown("### Rate this track")
        st.caption("Help Auralis learn your taste.")
        if st.button("👍  Like", key="like_f1", use_container_width=True):
            _apply_feedback(f1, e1, "like")
            st.success("Added to your profile!")
        if st.button("👎  Dislike", key="dislike_f1", use_container_width=True):
            _apply_feedback(f1, e1, "dislike")
            st.info("Got it — noted as a dislike.")

    # ── File 2 ──────────────────────────────────────────────────────────────
    if file2 is not None:
        with st.spinner("Analyzing File 2…"):
            path2 = save_temp_file(file2)
            f2 = extract_features(path2)
            e2 = map_emotion(f2.meta)

        col2_res, col2_act = st.columns([3, 1])
        with col2_res:
            st.markdown("### File 2 Results")
            st.write(f"**Predicted emotion:** `{e2.emotion}`")
            file_summary(f2)
            with st.expander("Emotion scores"):
                emotion_bar(e2.scores)
            with st.expander("Raw feature metadata"):
                st.json(f2.meta)

        with col2_act:
            st.markdown("### Rate this track")
            st.caption("Help Auralis learn your taste.")
            if st.button("👍  Like", key="like_f2", use_container_width=True):
                _apply_feedback(f2, e2, "like")
                st.success("Added to your profile!")
            if st.button("👎  Dislike", key="dislike_f2", use_container_width=True):
                _apply_feedback(f2, e2, "dislike")
                st.info("Got it — noted as a dislike.")

        # ── Similarity ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔍 Similarity Analysis")
        with st.spinner("Comparing…"):
            sim = compare_features(f1, f2)
            similarity = float(sim["similarity"])
        st.metric("Cosine similarity", round(similarity, 4))
        if similarity >= 0.90:
            st.caption("Very similar — close in MFCC feature space.")
        elif similarity >= 0.75:
            st.caption("Somewhat similar — shared traits, but not identical.")
        else:
            st.caption("Less similar — different timbre/texture/energy patterns.")
    else:
        st.markdown("---")
        st.info("Upload **Audio File 2** to compare similarity.")

    st.markdown("---")
    st.caption(
        "Note: Emotion mapping is rule-based and interpretable — "
        "a transparent baseline, not a clinical label."
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
elif tab_choice == "⭐ Recommendations":
    st.title("⭐ Recommendations")

    if not profile.has_signal():
        st.warning(
            "Your profile is empty. Go to **🎧 Analyze**, upload a track, "
            "and hit 👍 Like to teach Auralis your taste."
        )
        st.stop()

    # Load index
    try:
        index = load_index(INDEX_PATH)
    except FileNotFoundError:
        st.error(
            f"Song index not found at `{INDEX_PATH}`. "
            "Run `python -m tools.build_index` to generate it first."
        )
        st.stop()

    if len(index) == 0:
        st.warning("The song index is empty — add tracks and rebuild the index.")
        st.stop()

    # Controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        top_k = st.slider("Number of recommendations", 3, min(20, len(index)), 5)
    with col_ctrl2:
        alpha = st.slider(
            "Acoustic vs emotion weight",
            0.0, 1.0, 0.7, 0.05,
            help="1.0 = pure MFCC similarity · 0.0 = pure emotion match",
        )

    exclude = list(st.session_state.rated_paths) if st.checkbox("Hide already-rated tracks", value=True) else []

    with st.spinner("Ranking songs…"):
        try:
            recs = rank_songs(profile, index, alpha=alpha, exclude_paths=exclude, top_k=top_k)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if not recs:
        st.info("No unrated tracks left in the index. Uncheck 'Hide already-rated tracks' to see all.")
        st.stop()

    st.markdown(f"Showing top **{len(recs)}** tracks matched to your profile "
                f"(dominant taste: `{profile.dominant_emotion()}`).")
    st.markdown("---")

    for i, rec in enumerate(recs, 1):
        track_name = Path(rec["path"]).stem
        with st.expander(f"#{i} · {track_name}  —  {rec['emotion']}  ·  score {rec['blended_score']:.2f}"):
            st.caption(rec["explanation"])
            col_s, col_e = st.columns(2)
            col_s.metric("Acoustic match", f"{rec['mfcc_sim']:.0%}")
            col_e.metric("Emotion match",  f"{rec['emotion_sim']:.0%}")

            import pandas as pd
            scores_df = pd.DataFrame(
                {"Emotion": list(rec["emotion_scores"].keys()),
                 "Score":   list(rec["emotion_scores"].values())}
            ).set_index("Emotion")
            st.bar_chart(scores_df)

            # Feedback buttons inside recommendation card
            fb_col1, fb_col2, _ = st.columns([1, 1, 4])
            if fb_col1.button("👍", key=f"rec_like_{i}"):
                # We only have meta from index, not a live FeatureOutput —
                # build a minimal profile update from stored emotion scores
                import numpy as np
                dummy_vec = np.zeros(26)   # no stored vector in current index schema
                profile = record_feedback(
                    profile=st.session_state.profile,
                    vector=dummy_vec,
                    emotion_scores=rec["emotion_scores"],
                    emotion_label=rec["emotion"],
                    track_path=rec["path"],
                    label="like",
                    profile_save_path=PROFILE_PATH,
                )
                st.session_state.profile = profile
                st.session_state.rated_paths.add(rec["path"])
                st.success("Profile updated!")

            if fb_col2.button("👎", key=f"rec_dislike_{i}"):
                import numpy as np
                dummy_vec = np.zeros(26)
                profile = record_feedback(
                    profile=st.session_state.profile,
                    vector=dummy_vec,
                    emotion_scores=rec["emotion_scores"],
                    emotion_label=rec["emotion"],
                    track_path=rec["path"],
                    label="dislike",
                    profile_save_path=PROFILE_PATH,
                )
                st.session_state.profile = profile
                st.session_state.rated_paths.add(rec["path"])
                st.info("Noted.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MY PROFILE
# ════════════════════════════════════════════════════════════════════════════
elif tab_choice == "👤 My Profile":
    st.title("👤 My Preference Profile")

    if not profile.has_signal():
        st.info("No data yet. Rate some tracks in **🎧 Analyze** or **⭐ Recommendations**.")
        st.stop()

    summary = feedback_summary(profile)

    # ── Overview metrics ────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Liked tracks",    summary["total_likes"])
    m2.metric("Disliked tracks", summary["total_dislikes"])
    m3.metric("Total ratings",   summary["interactions"])
    m4.metric("Dominant emotion", summary["dominant_emotion"] or "—")

    # ── Emotion affinity chart ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Emotion Affinity")
    st.caption(
        "Your accumulated preference across emotional dimensions. "
        "Derived from all tracks you've liked so far."
    )
    import pandas as pd
    aff_df = pd.DataFrame(
        {"Emotion": list(summary["emotion_affinity"].keys()),
         "Affinity": list(summary["emotion_affinity"].values())}
    ).set_index("Emotion")
    st.bar_chart(aff_df)

    # ── Interaction log ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Interaction History")
    if profile.interaction_log:
        log_df = pd.DataFrame(profile.interaction_log)
        log_df["track"] = log_df["track"].apply(lambda p: Path(p).name)
        st.dataframe(log_df[["track", "feedback", "emotion_label"]], use_container_width=True)
    else:
        st.info("No interactions logged yet.")

    # ── Reset ────────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🗑️ Reset my profile", type="secondary"):
        st.session_state.profile = UserProfile()
        st.session_state.rated_paths = set()
        UserProfile().save(PROFILE_PATH)
        st.success("Profile reset.")
        st.rerun()
