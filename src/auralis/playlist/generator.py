"""
Auralis - Playlist Generator
------------------------------
Generates emotion-aware playlists from the indexed song collection.

Two generation modes:
  1. Emotion filter  — rank songs by a specific target emotion score
  2. Profile-based   — rank songs against the user's preference profile

Both modes return a ranked, deduplicated list of tracks with full
emotion metadata, ready for display or CSV export.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from src.auralis.preference.profile import UserProfile
from src.auralis.preference.recommender import _cosine

EMOTIONS = ["calm", "energetic", "happy", "sad"]


def _score_by_emotion(row: Dict, target_emotion: str) -> float:
    """Score a track by how strongly it matches a target emotion."""
    try:
        return float(row.get(target_emotion, 0.0))
    except (ValueError, TypeError):
        return 0.0


def _score_by_profile(row: Dict, profile: UserProfile) -> float:
    """Score a track against the user's emotion affinity profile."""
    pref = profile.emotion_affinity
    pref_vec   = np.array([pref.get(e, 0.0) for e in EMOTIONS])
    track_vec  = np.array([
        float(row.get(e, 0.0)) for e in EMOTIONS
    ])
    return max(0.0, _cosine(pref_vec, track_vec))


def generate_playlist(
    index: List[Dict],
    mode: str = "emotion",           # "emotion" | "profile"
    target_emotion: Optional[str] = "calm",
    profile: Optional[UserProfile] = None,
    length: int = 8,
    exclude_paths: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Generate a playlist from the indexed song collection.

    Parameters
    ----------
    index          : list of track dicts from research_index.csv
    mode           : "emotion" to filter by target_emotion,
                     "profile" to match user preference profile
    target_emotion : emotion label to target (used when mode="emotion")
    profile        : UserProfile (required when mode="profile")
    length         : number of tracks in the playlist
    exclude_paths  : paths to skip

    Returns
    -------
    List of track dicts sorted by relevance score, each containing:
      rank, track_name, path, dominant_emotion, emotion_scores,
      relevance_score, mode
    """
    if mode == "profile" and (profile is None or not profile.has_signal()):
        raise ValueError("Profile mode requires a profile with at least one liked track.")

    exclude = set(exclude_paths or [])
    scored = []

    for row in index:
        path = row.get("path", "")
        if path in exclude:
            continue

        if mode == "emotion":
            score = _score_by_emotion(row, target_emotion)
        else:
            score = _score_by_profile(row, profile)

        emotion_scores = {}
        for e in EMOTIONS:
            try:
                emotion_scores[e] = round(float(row.get(e, 0.0)), 4)
            except (ValueError, TypeError):
                emotion_scores[e] = 0.0

        dominant = row.get("predicted_emotion") or row.get("emotion", "unknown")
        track_name = Path(path).stem

        scored.append({
            "track_name":     track_name,
            "path":           path,
            "dominant_emotion": dominant,
            "emotion_scores": emotion_scores,
            "relevance_score": round(score, 4),
            "mode":           mode,
            "target":         target_emotion if mode == "emotion" else "profile",
        })

    # Sort by relevance descending
    scored.sort(key=lambda r: r["relevance_score"], reverse=True)
    playlist = scored[:length]

    # Add rank
    for i, track in enumerate(playlist, 1):
        track["rank"] = i

    return playlist


def playlist_to_csv(playlist: List[Dict]) -> str:
    """
    Serialize a playlist to a CSV string for download.

    Columns: rank, track_name, dominant_emotion, calm, energetic, happy, sad,
             relevance_score, mode, target, path
    """
    output = io.StringIO()
    fieldnames = [
        "rank", "track_name", "dominant_emotion",
        "calm", "energetic", "happy", "sad",
        "relevance_score", "mode", "target", "path"
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for track in playlist:
        scores = track.get("emotion_scores", {})
        writer.writerow({
            "rank":             track["rank"],
            "track_name":       track["track_name"],
            "dominant_emotion": track["dominant_emotion"],
            "calm":             scores.get("calm", 0.0),
            "energetic":        scores.get("energetic", 0.0),
            "happy":            scores.get("happy", 0.0),
            "sad":              scores.get("sad", 0.0),
            "relevance_score":  track["relevance_score"],
            "mode":             track["mode"],
            "target":           track["target"],
            "path":             track["path"],
        })

    return output.getvalue()
