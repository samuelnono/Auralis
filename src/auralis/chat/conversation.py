"""
Auralis - Conversation Manager
--------------------------------
Builds a grounded system prompt from the user's live profile and
song index so the LLM always reasons from real acoustic data,
never from hallucinated music knowledge.
"""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from src.auralis.preference.profile import UserProfile


def build_system_prompt(
    profile: UserProfile,
    index: Optional[List[Dict]] = None,
    last_analyzed_track: Optional[Dict] = None,
) -> str:
    """
    Build a fully grounded system prompt for the Auralis chat assistant.

    Injects:
      - User's emotion affinity profile
      - Dominant taste and rating history summary
      - Top 5 indexed songs (name + emotion) for recommendation context
      - Last analyzed track metadata (if available)
    """

    # ── Profile summary ──────────────────────────────────────────────────────
    if profile.has_signal():
        affinity = profile.emotion_affinity
        dominant = profile.dominant_emotion()
        affinity_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in sorted(affinity.items(), key=lambda x: -x[1])
        )
        profile_block = (
            f"The user has rated {profile.total_likes} track(s) as liked "
            f"and {profile.total_dislikes} as disliked.\n"
            f"Their dominant emotion preference is: {dominant}.\n"
            f"Emotion affinity scores (0-1): {affinity_str}."
        )
    else:
        profile_block = "The user has not rated any tracks yet. No preference profile exists."

    # ── Index summary (top 8 tracks by name + emotion + mood) ────────────────
    if index:
        track_lines = []
        for row in index[:8]:
            from pathlib import Path
            name = Path(row.get("path", "unknown")).stem
            emotion = row.get("predicted_emotion") or row.get("emotion", "unknown")
            calm      = float(row.get("calm", 0)      or 0)
            energetic = float(row.get("energetic", 0) or 0)
            happy     = float(row.get("happy", 0)     or 0)
            sad       = float(row.get("sad", 0)       or 0)

            # Optional continuous-mood coordinates (populated by v1.1+ indexer).
            mood_suffix = ""
            try:
                v = float(row["valence"])
                a = float(row["arousal"])
                q = row.get("quadrant", "")
                n = row.get("nuanced_tag", "")
                mood_suffix = (
                    f", mood=({v:+.2f}, {a:+.2f}) [{q or 'unknown'}"
                    + (f" / {n}" if n else "")
                    + "]"
                )
            except (KeyError, TypeError, ValueError):
                pass

            track_lines.append(
                f"  - {name}: dominant={emotion}, "
                f"calm={calm:.2f}, energetic={energetic:.2f}, "
                f"happy={happy:.2f}, sad={sad:.2f}{mood_suffix}"
            )
        index_block = "Available indexed tracks:\n" + "\n".join(track_lines)
    else:
        index_block = "No indexed tracks are available yet."

    # ── Last analyzed track ──────────────────────────────────────────────────
    if last_analyzed_track:
        from pathlib import Path
        name = Path(last_analyzed_track.get("path", "unknown")).stem
        emotion = last_analyzed_track.get("predicted_emotion") or last_analyzed_track.get("emotion", "unknown")
        tempo   = last_analyzed_track.get("tempo", "N/A")
        rms     = last_analyzed_track.get("rms_mean", "N/A")
        centroid = last_analyzed_track.get("spectral_centroid_mean", "N/A")
        duration = last_analyzed_track.get("duration_sec", "N/A")

        mood = last_analyzed_track.get("mood") or {}
        if mood:
            mood_line = (
                f"  Dimensional mood: valence={mood.get('valence', 0):+.2f}, "
                f"arousal={mood.get('arousal', 0):+.2f}, "
                f"quadrant={mood.get('quadrant', 'unknown')}, "
                f"nuanced={mood.get('nuanced_tag', 'unknown')}"
            )
        else:
            mood_line = "  Dimensional mood: not available"

        track_block = (
            f"The user most recently analyzed a track called '{name}'.\n"
            f"  Predicted emotion (discrete): {emotion}\n"
            f"  Tempo: {round(float(tempo), 1) if tempo != 'N/A' else 'N/A'} BPM\n"
            f"  RMS energy: {round(float(rms), 4) if rms != 'N/A' else 'N/A'}\n"
            f"  Spectral centroid: {round(float(centroid), 1) if centroid != 'N/A' else 'N/A'} Hz\n"
            f"  Duration: {round(float(duration), 1) if duration != 'N/A' else 'N/A'} sec\n"
            f"{mood_line}"
        )
    else:
        track_block = "No track has been analyzed in this session yet."

    # ── Full system prompt ───────────────────────────────────────────────────
    return f"""You are Auralis, an intelligent music assistant that combines acoustic signal processing with emotional awareness to help users discover and understand music.

You are grounded in real data — all your responses about tracks, emotions, and recommendations must reference the acoustic features and profile data provided below. Do not invent music knowledge or pretend to know songs you haven't been given data about.

## User Preference Profile
{profile_block}

## Song Index
{index_block}

## Recently Analyzed Track
{track_block}

## Your Capabilities
1. **Recommend songs** from the indexed collection based on the user's mood request, matching against their preference profile and the available emotion scores.
2. **Explain recommendations** by referencing actual acoustic features: tempo, RMS energy, spectral centroid, MFCC-derived emotion scores, and dimensional mood coordinates.
3. **Answer questions** about any analyzed track's emotional and acoustic properties, including where it sits on the valence-arousal circumplex.
4. **Discuss music taste** freely, using the user's emotion affinity profile as context.

## Emotional Representation
Auralis represents mood on two complementary layers:
- **Discrete (4-class):** calm / energetic / happy / sad — the interpretable baseline.
- **Dimensional (continuous):** valence (gloomy → uplifting) and arousal (calm → energetic)
  coordinates in [-1, 1], combined with a quadrant label (Excited / Tense / Melancholic / Serene)
  and a nuanced sub-tag (Uplifting, Triumphant, Intense, Tense, Gloomy, Melancholic, Reflective, Peaceful).
When both layers are available, reference whichever best answers the user's question — the
dimensional layer is more precise for subtle distinctions, the discrete layer is more readable.

## Rules
- Always ground explanations in the acoustic data provided. Say "based on its spectral centroid of X Hz" not "this song sounds bright".
- If asked about a song not in the index, say you don't have acoustic data for it yet.
- Keep responses concise and conversational. Use plain language — translate technical terms when helpful.
- Never fabricate feature values, emotion scores, or mood coordinates.
"""


def format_history_for_api(history: List[Dict]) -> List[Dict]:
    """
    Convert Auralis chat history format to Anthropic API messages format.
    history items: {"role": "user"|"assistant", "content": str}
    """
    return [{"role": item["role"], "content": item["content"]} for item in history]
