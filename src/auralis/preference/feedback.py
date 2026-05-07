"""
Auralis - Feedback Handler
---------------------------
Thin layer that wires a UI feedback event (like / dislike) to a
UserProfile update. Keeps app.py clean by handling all the type
coercion and persistence in one place.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

from .profile import UserProfile


def record_feedback(
    profile: UserProfile,
    vector: np.ndarray,
    emotion_scores: Dict[str, float],
    emotion_label: str,
    track_path: str,
    label: str,                          # "like" | "dislike"
    profile_save_path: str,              # where to persist the updated profile
    display_name: str | None = None,     # original upload filename, for the UI
) -> UserProfile:
    """
    Apply one feedback event to the profile and persist it.

    Returns the updated profile so callers can chain or reassign.
    """
    if label not in ("like", "dislike"):
        raise ValueError(f"label must be 'like' or 'dislike', got {label!r}")

    profile.apply_feedback(
        vector=vector,
        emotion_scores=emotion_scores,
        label=label,
        track_path=track_path,
        emotion_label=emotion_label,
        display_name=display_name,
    )
    profile.save(profile_save_path)
    return profile


def record_spotify_feedback(
    profile: UserProfile,
    spotify_id: str,
    track_name: str,
    artists: list[str],
    label: str,                       # "like" | "dislike"
    emotion_scores: Dict[str, float], # synthetic, derived from browsed mood
    emotion_label: str,               # "calm" | "energetic" | "happy" | "sad"
    profile_save_path: str,
) -> UserProfile:
    """
    Record a like/dislike on a *Spotify* track (rather than a locally-uploaded
    audio file). Spotify tracks don't come with an MFCC feature vector — we
    only have the track metadata and the mood the user was browsing — so we
    skip the preference_vector update and only nudge the emotion_affinity
    counters and append to interaction_log.

    The interaction_log entry gets a stable ``track`` key (``spotify:track:ID``)
    and a ``display_name`` of ``"Track Name - Primary Artist"`` so the Profile
    page's Interaction History renders it cleanly alongside locally-rated
    uploads.
    """
    if label not in ("like", "dislike"):
        raise ValueError(f"label must be 'like' or 'dislike', got {label!r}")

    primary_artist = (artists[0] if artists else "").strip()
    display_name = f"{track_name} — {primary_artist}" if primary_artist else track_name
    track_uri = f"spotify:track:{spotify_id}" if spotify_id else display_name

    if label == "like":
        profile.total_likes += 1
        n = profile.total_likes
        # Pull the emotion-affinity centroid toward the mood scores, just
        # like apply_feedback does for local tracks. We deliberately do NOT
        # update preference_vector since we have no MFCC for Spotify tracks.
        for emotion, score in emotion_scores.items():
            old = profile.emotion_affinity.get(emotion, 0.0)
            profile.emotion_affinity[emotion] = old + (score - old) / n
    else:
        profile.total_dislikes += 1

    profile.interaction_log.append(
        {
            "track": track_uri,
            "display_name": display_name,
            "feedback": label,
            "emotion_label": emotion_label,
            "emotion_scores": {k: round(v, 4) for k, v in emotion_scores.items()},
            "source": "spotify",
        }
    )

    profile.save(profile_save_path)
    return profile


def feedback_summary(profile: UserProfile) -> Dict:
    """
    Returns a compact summary dict for display in the UI.
    """
    return {
        "total_likes": profile.total_likes,
        "total_dislikes": profile.total_dislikes,
        "dominant_emotion": profile.dominant_emotion(),
        "emotion_affinity": {
            k: round(v, 3) for k, v in profile.emotion_affinity.items()
        },
        "interactions": len(profile.interaction_log),
    }
