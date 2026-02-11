"""
Auralis - Phase 3A: Rule-based emotion mapping (interpretable)

We map audio features -> emotion scores using simple, human-readable rules.
This is intentionally transparent so you can explain it in a paper/demo.

Current inputs expected (from FeatureOutput.meta / features.py):
- mfcc_mean: np.ndarray shape (n_mfcc,)
- mfcc_std:  np.ndarray shape (n_mfcc,)
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass(frozen=True)
class EmotionOutput:
    emotion: str
    scores: Dict[str, float]
    meta: Dict[str, Any]


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _z(x: float, lo: float, hi: float) -> float:
    """
    Normalize x into [0,1] using a rough expected range [lo, hi].
    Values outside are clamped. This keeps rules stable across files.
    """
    if hi <= lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))


def map_emotion(features_meta: Dict[str, Any]) -> EmotionOutput:
    """
    Rule-based mapping from aggregated audio features to an emotion label.

    Input:
      features_meta: dict that contains (at minimum) mfcc_mean and mfcc_std arrays.

    Output:
      EmotionOutput with:
        - emotion: top label
        - scores: interpretable scores for each label (0..1)
        - meta: useful debug info for reporting
    """
    mfcc_mean = np.asarray(features_meta.get("mfcc_mean", []), dtype=float)
    mfcc_std = np.asarray(features_meta.get("mfcc_std", []), dtype=float)

    if mfcc_mean.size == 0 or mfcc_std.size == 0:
        raise ValueError("features_meta must include non-empty 'mfcc_mean' and 'mfcc_std'.")

    # --- Interpretable proxies (simple + explainable) ---
    # Variability proxy: overall spectral change/complexity
    variability = float(np.mean(mfcc_std))  # often ~10-80 depending on audio
    # Brightness proxy: higher MFCC indices tend to relate to finer spectral detail
    # Using mean of MFCC 6-12 (0-indexed slice 5:13) as a rough "brightness" signal
    hi_band = float(np.mean(mfcc_mean[5:13])) if mfcc_mean.size >= 13 else float(np.mean(mfcc_mean))
    # Warmth proxy: low MFCCs relate more to overall spectral envelope / low-frequency energy
    lo_band = float(np.mean(mfcc_mean[0:5])) if mfcc_mean.size >= 5 else float(np.mean(mfcc_mean))

    # --- Normalize into 0..1 ranges (rough expected ranges) ---
    v = _z(variability, lo=5.0, hi=80.0)
    bright = _z(hi_band, lo=-50.0, hi=200.0)
    warm = _z(lo_band, lo=-350.0, hi=50.0)

    # --- Emotion rules (simple, readable) ---
    # We keep 4 emotions for now (easy to defend in a paper + demo):
    # - energetic: high variability + higher brightness
    # - calm: low variability + warmer (less bright)
    # - happy: moderate-high brightness + moderate variability
    # - sad: low brightness + moderate warmth + lower variability
    energetic = _clamp01(0.60 * v + 0.40 * bright)
    calm = _clamp01(0.65 * (1.0 - v) + 0.35 * warm)
    happy = _clamp01(0.55 * bright + 0.45 * _clamp01(1.0 - abs(v - 0.55)))
    sad = _clamp01(0.60 * (1.0 - bright) + 0.40 * (1.0 - v))

    scores = {
        "energetic": energetic,
        "calm": calm,
        "happy": happy,
        "sad": sad,
    }

    top_emotion = max(scores, key=scores.get)

    meta = {
        "variability_mean_mfcc_std": variability,
        "hi_band_mfcc_mean": hi_band,
        "lo_band_mfcc_mean": lo_band,
        "normalized": {"v": v, "bright": bright, "warm": warm},
        "source_path": features_meta.get("path"),
        "duration_sec": features_meta.get("duration_sec"),
        "sr": features_meta.get("sr"),
    }

    return EmotionOutput(emotion=top_emotion, scores=scores, meta=meta)
