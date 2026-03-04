"""
Auralis - Emotion Mapping (v2)
-------------------------------
Rule-based emotion mapping upgraded to use the richer feature set:
  - 20 MFCCs  (expanded from 13)
  - Tempo     (BPM)
  - RMS energy
  - Spectral centroid

Emotion model still targets 4 interpretable labels aligned with the
valence-arousal framework: calm, energetic, happy, sad.

All rules remain transparent and explainable for research / demo use.
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
    """Normalise x into [0, 1] using expected range [lo, hi]."""
    if hi <= lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))


def map_emotion(features_meta: Dict[str, Any]) -> EmotionOutput:
    """
    Rule-based mapping from aggregated audio features to an emotion label.

    Inputs (from FeatureOutput.meta):
      mfcc_mean               – list/array of 20 MFCC means
      mfcc_std                – list/array of 20 MFCC stds
      tempo                   – BPM float
      rms_mean                – mean RMS energy float
      spectral_centroid_mean  – mean spectral centroid in Hz

    Output:
      EmotionOutput with emotion label, scores dict, and debug meta.
    """
    mfcc_mean = np.asarray(features_meta.get("mfcc_mean", []), dtype=float)
    mfcc_std  = np.asarray(features_meta.get("mfcc_std",  []), dtype=float)

    if mfcc_mean.size == 0 or mfcc_std.size == 0:
        raise ValueError("features_meta must include non-empty 'mfcc_mean' and 'mfcc_std'.")

    # ── Acoustic proxies ─────────────────────────────────────────────────────

    # Spectral variability — overall complexity / movement
    variability = float(np.mean(mfcc_std))

    # Timbral brightness — higher MFCC indices (fine spectral detail)
    hi_band = float(np.mean(mfcc_mean[6:20])) if mfcc_mean.size >= 13 else float(np.mean(mfcc_mean))

    # Warmth — low MFCC indices (spectral envelope / low-freq energy)
    lo_band = float(np.mean(mfcc_mean[0:6])) if mfcc_mean.size >= 6 else float(np.mean(mfcc_mean))

    # Tempo — rhythmic energy (BPM)
    tempo = float(features_meta.get("tempo", 120.0))

    # RMS energy — loudness / intensity
    rms = float(features_meta.get("rms_mean", 0.05))

    # Spectral centroid — perceived brightness in Hz
    centroid = float(features_meta.get("spectral_centroid_mean", 2000.0))

    # ── Normalise to [0, 1] ──────────────────────────────────────────────────
    v       = _z(variability, lo=5.0,    hi=80.0)
    bright  = _z(hi_band,     lo=-30.0,  hi=150.0)
    warm    = _z(lo_band,     lo=-350.0, hi=50.0)
    t       = _z(tempo,       lo=60.0,   hi=180.0)   # 60 BPM=calm, 180=energetic
    e       = _z(rms,         lo=0.0,    hi=0.3)     # energy proxy
    c       = _z(centroid,    lo=500.0,  hi=6000.0)  # brightness in Hz

    # ── Emotion rules ────────────────────────────────────────────────────────
    #
    # energetic : high tempo + high energy + spectral brightness
    energetic = _clamp01(0.35 * t + 0.35 * e + 0.20 * v + 0.10 * c)

    # calm      : low tempo + low energy + low variability + warmth
    calm      = _clamp01(0.35 * (1.0 - t) + 0.30 * (1.0 - e) + 0.20 * (1.0 - v) + 0.15 * warm)

    # happy     : bright timbre + moderate-high tempo + moderate energy
    happy     = _clamp01(0.35 * c + 0.30 * bright + 0.20 * t + 0.15 * _clamp01(1.0 - abs(e - 0.5)))

    # sad       : low brightness + low tempo + lower energy
    sad       = _clamp01(0.35 * (1.0 - c) + 0.30 * (1.0 - t) + 0.25 * (1.0 - bright) + 0.10 * (1.0 - e))

    scores = {
        "energetic": energetic,
        "calm":      calm,
        "happy":     happy,
        "sad":       sad,
    }

    top_emotion = max(scores, key=scores.get)

    meta = {
        "variability_mean_mfcc_std":   variability,
        "hi_band_mfcc_mean":           hi_band,
        "lo_band_mfcc_mean":           lo_band,
        "tempo_bpm":                   tempo,
        "rms_mean":                    rms,
        "spectral_centroid_mean_hz":   centroid,
        "normalized": {
            "v": v, "bright": bright, "warm": warm,
            "t": t, "e": e,  "c": c,
        },
        "source_path":  features_meta.get("path"),
        "duration_sec": features_meta.get("duration_sec"),
        "sr":           features_meta.get("sr"),
    }

    return EmotionOutput(emotion=top_emotion, scores=scores, meta=meta)
