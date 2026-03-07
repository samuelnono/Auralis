"""
Auralis - User Preference Profile
----------------------------------
Maintains a running preference vector in MFCC feature space.
The profile is a weighted centroid of liked tracks, updated
incrementally so it never needs to store raw audio.

Design goals:
  - Transparent: every update is traceable
  - Lightweight: no external DB; serializes to JSON
  - Cold-start safe: works from the very first rating
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


EMOTIONS = ["energetic", "calm", "happy", "sad"]


@dataclass
class UserProfile:
    """
    Stores a user's accumulated preference signal.

    preference_vector : weighted mean of liked MFCC vectors (dim=46)
    emotion_affinity  : weighted mean of liked emotion scores (0-1 per label)
    interaction_log   : list of every feedback event for auditability
    total_likes       : used for incremental centroid update
    total_dislikes    : informational
    """

    preference_vector: List[float] = field(default_factory=list)
    emotion_affinity: Dict[str, float] = field(
        default_factory=lambda: {e: 0.0 for e in EMOTIONS}
    )
    interaction_log: List[Dict] = field(default_factory=list)
    total_likes: int = 0
    total_dislikes: int = 0

    # ------------------------------------------------------------------
    # Core update logic
    # ------------------------------------------------------------------

    def apply_feedback(
        self,
        vector: np.ndarray,
        emotion_scores: Dict[str, float],
        label: str,       # "like" | "dislike"
        track_path: str,
        emotion_label: str,
    ) -> None:
        """
        Update the preference profile given user feedback on one track.

        Likes    → pull the preference vector toward this track (incremental mean).
        Dislikes → push the preference vector away (small penalty step).
        """
        vec = np.asarray(vector, dtype=float)

        if label == "like":
            self.total_likes += 1
            n = self.total_likes

            if len(self.preference_vector) == 0:
                # First like — initialise directly
                self.preference_vector = vec.tolist()
                self.emotion_affinity = {k: float(v) for k, v in emotion_scores.items()}
            else:
                pv = np.asarray(self.preference_vector, dtype=float)

                # Dimension mismatch — happens after a feature upgrade (e.g. 26→46 dims).
                # Reset the profile vector to the new track rather than crashing.
                if pv.shape != vec.shape:
                    self.preference_vector = vec.tolist()
                    self.total_likes = 1
                    n = 1
                else:
                    # Incremental centroid: new_mean = old_mean + (x - old_mean) / n
                    pv = pv + (vec - pv) / n
                    self.preference_vector = pv.tolist()

                for emotion in EMOTIONS:
                    old = self.emotion_affinity.get(emotion, 0.0)
                    self.emotion_affinity[emotion] = old + (
                        emotion_scores.get(emotion, 0.0) - old
                    ) / n

        elif label == "dislike":
            self.total_dislikes += 1

            if len(self.preference_vector) > 0:
                pv = np.asarray(self.preference_vector, dtype=float)

                # Skip nudge if dims don't match
                if pv.shape == vec.shape:
                    direction = vec - pv
                    norm = np.linalg.norm(direction)
                    if norm > 0:
                        pv = pv - 0.05 * direction / norm
                        self.preference_vector = pv.tolist()

        # Always log
        self.interaction_log.append(
            {
                "track": track_path,
                "feedback": label,
                "emotion_label": emotion_label,
                "emotion_scores": {k: round(v, 4) for k, v in emotion_scores.items()},
            }
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_signal(self) -> bool:
        """True once the user has liked at least one track."""
        return self.total_likes > 0

    def preference_np(self) -> Optional[np.ndarray]:
        if not self.preference_vector:
            return None
        return np.asarray(self.preference_vector, dtype=float)

    def dominant_emotion(self) -> Optional[str]:
        if not self.has_signal():
            return None
        return max(self.emotion_affinity, key=self.emotion_affinity.get)

    # ------------------------------------------------------------------
    # Persistence (JSON — no external deps)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "UserProfile":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def load_or_new(cls, path: str) -> "UserProfile":
        try:
            return cls.load(path)
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()
