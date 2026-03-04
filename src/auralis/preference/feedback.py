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
    label: str,              # "like" | "dislike"
    profile_save_path: str,  # where to persist the updated profile
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
