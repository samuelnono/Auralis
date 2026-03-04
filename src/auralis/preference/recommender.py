"""
Auralis - Preference-Aware Recommender
----------------------------------------
Ranks an indexed song collection against a UserProfile using a
blended score of:
  1. MFCC cosine similarity  (acoustic proximity to liked tracks)
  2. Emotion affinity score  (alignment with preferred emotion distribution)

Both signals are interpretable and directly traceable to user feedback.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from .profile import UserProfile

EMOTIONS = ["energetic", "calm", "happy", "sad"]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_vector(raw) -> Optional[np.ndarray]:
    """Parse a stored MFCC vector from CSV (stored as JSON list string)."""
    if raw is None or raw == "":
        return None
    try:
        return np.asarray(json.loads(raw), dtype=float)
    except (json.JSONDecodeError, ValueError):
        return None


def load_index(index_path: str) -> List[Dict]:
    """
    Load the research_index.csv produced by tools/build_index.py.
    Expected columns (minimum):
      path, emotion, calm, energetic, happy, sad, mfcc_vector (JSON list)
    Returns a list of dicts, one per track.
    """
    rows = []
    with open(index_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def rank_songs(
    profile: UserProfile,
    index: List[Dict],
    alpha: float = 0.7,
    exclude_paths: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict]:
    """
    Rank indexed songs against the user's preference profile.

    Parameters
    ----------
    profile      : UserProfile — the user's accumulated preference signal
    index        : list of track dicts loaded from the CSV index
    alpha        : weight for MFCC similarity (1-alpha goes to emotion affinity)
    exclude_paths: paths to skip (e.g. already-rated tracks)
    top_k        : number of results to return

    Returns
    -------
    List of dicts sorted by blended_score descending, each containing:
      path, emotion, emotion_scores, mfcc_sim, emotion_sim, blended_score, explanation
    """
    if not profile.has_signal():
        raise ValueError("Profile has no signal yet. User must like at least one track.")

    pref_vec = profile.preference_np()
    pref_emotion = profile.emotion_affinity
    exclude = set(exclude_paths or [])

    results = []

    for row in index:
        path = row.get("path", "")
        if path in exclude:
            continue

        # --- MFCC similarity ---
        raw_vec = row.get("mfcc_vector", None)
        track_vec = _parse_vector(raw_vec)
        if track_vec is None or pref_vec is None:
            mfcc_sim = 0.0
        else:
            mfcc_sim = max(0.0, _cosine(pref_vec, track_vec))

        # --- Emotion affinity similarity ---
        track_emotions = {}
        for e in EMOTIONS:
            try:
                track_emotions[e] = float(row.get(e, 0.0))
            except (ValueError, TypeError):
                track_emotions[e] = 0.0

        # Cosine similarity in emotion score space
        pref_e_vec = np.array([pref_emotion.get(e, 0.0) for e in EMOTIONS])
        track_e_vec = np.array([track_emotions.get(e, 0.0) for e in EMOTIONS])
        emotion_sim = max(0.0, _cosine(pref_e_vec, track_e_vec))

        # --- Blended score ---
        blended = alpha * mfcc_sim + (1.0 - alpha) * emotion_sim

        # --- Human-readable explanation ---
        dominant_user = profile.dominant_emotion() or "unknown"
        dominant_track = row.get("emotion", "unknown")
        explanation = (
            f"Acoustic match: {mfcc_sim:.0%} · "
            f"Emotion match: {emotion_sim:.0%} · "
            f"Your taste leans {dominant_user}; this track is {dominant_track}."
        )

        results.append(
            {
                "path": path,
                "emotion": dominant_track,
                "emotion_scores": track_emotions,
                "mfcc_sim": round(mfcc_sim, 4),
                "emotion_sim": round(emotion_sim, 4),
                "blended_score": round(blended, 4),
                "explanation": explanation,
            }
        )

    results.sort(key=lambda r: r["blended_score"], reverse=True)
    return results[:top_k]
