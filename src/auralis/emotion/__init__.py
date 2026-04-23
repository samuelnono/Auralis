"""
Auralis - Emotion package
-------------------------
Exports:
  - map_emotion, EmotionOutput            (rule-based 4-class baseline)
  - compute_mood, mood_from_discrete, MoodPoint
                                          (continuous valence-arousal layer)
"""

from .emotion import map_emotion, EmotionOutput
from .valence_arousal import compute_mood, mood_from_discrete, MoodPoint

__all__ = [
    "map_emotion",
    "EmotionOutput",
    "compute_mood",
    "mood_from_discrete",
    "MoodPoint",
]
