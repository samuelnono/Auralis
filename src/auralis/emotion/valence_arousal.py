"""
Auralis — Valence-Arousal Dimensional Emotion Layer (v1.1)
-----------------------------------------------------------
Upgrades the discrete 4-class emotion output (calm / energetic / happy / sad)
with a continuous mood representation on Russell's circumplex model:

                        arousal (energy)
                              ▲
                              │
                Tense  Q2 ────┼──── Q1  Excited
                              │
            valence ◄─────────┼─────────► valence
                              │
          Melancholic Q3 ─────┼──── Q4  Serene
                              │
                              ▼

Produces for every track:
  • valence     ∈ [-1, 1]  (negative = gloomy, positive = uplifting)
  • arousal     ∈ [-1, 1]  (negative = calm,  positive = energetic)
  • radius      ∈ [0, √2]  (0 = emotionally neutral, higher = more intense)
  • angle_deg   ∈ [0, 360) (position on the mood wheel, 0°=east, ccw+)
  • quadrant    — one of 4 macro labels (Excited / Tense / Melancholic / Serene)
  • nuanced_tag — one of 8 fine-grained labels arranged around the wheel
  • confidence  — radius / √2, a proxy for how emotionally "committed" the track is
  • rationale   — human-readable explanation grounded in the acoustic features

The mapping reuses the normalised proxies (tempo, RMS, spectral centroid,
MFCC variability, brightness, warmth) already computed in emotion.py, so it
requires no extra feature extraction step. It is kept as a separate module
so the transparent 4-class rule-based baseline remains intact for research /
publication purposes — the two layers complement each other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Nuanced mood labels arranged counter-clockwise around the circumplex.
# Each covers a 45° arc. 0° points east (high valence, neutral arousal) and
# increases counter-clockwise so 90° is straight up (neutral valence, max arousal).
# ──────────────────────────────────────────────────────────────────────────────
_NUANCED_TAGS = (
    # (arc_start, arc_end, label)
    (337.5, 22.5,  "Uplifting"),    # high valence, ~neutral arousal
    ( 22.5, 67.5,  "Triumphant"),   # high valence + high arousal  (Q1 interior)
    ( 67.5, 112.5, "Intense"),      # neutral valence, high arousal
    (112.5, 157.5, "Tense"),        # low valence + high arousal   (Q2 interior)
    (157.5, 202.5, "Gloomy"),       # low valence, ~neutral arousal
    (202.5, 247.5, "Melancholic"),  # low valence + low arousal    (Q3 interior)
    (247.5, 292.5, "Reflective"),   # neutral valence, low arousal
    (292.5, 337.5, "Peaceful"),     # high valence + low arousal   (Q4 interior)
)


@dataclass(frozen=True)
class MoodPoint:
    """Dimensional mood coordinates + interpretable labels for one track."""

    valence: float       # [-1, 1]
    arousal: float       # [-1, 1]
    radius: float        # [0, √2]
    angle_deg: float     # [0, 360)
    quadrant: str        # Excited / Tense / Melancholic / Serene
    nuanced_tag: str     # one of _NUANCED_TAGS
    confidence: float    # radius / √2 ∈ [0, 1]
    rationale: str       # human-readable explanation

    def as_dict(self) -> Dict[str, Any]:
        return {
            "valence":     self.valence,
            "arousal":     self.arousal,
            "radius":      self.radius,
            "angle_deg":   self.angle_deg,
            "quadrant":    self.quadrant,
            "nuanced_tag": self.nuanced_tag,
            "confidence":  self.confidence,
            "rationale":   self.rationale,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _clamp01(x: float) -> float:
    return _clamp(x, 0.0, 1.0)


def _z(x: float, lo: float, hi: float) -> float:
    """Normalise x into [0, 1] using expected range [lo, hi]."""
    if hi <= lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))


def _quadrant(valence: float, arousal: float) -> str:
    if valence >= 0 and arousal >= 0:
        return "Excited"      # Q1
    if valence <  0 and arousal >= 0:
        return "Tense"        # Q2
    if valence <  0 and arousal <  0:
        return "Melancholic"  # Q3
    return "Serene"           # Q4 (valence ≥ 0, arousal < 0)


def _nuanced_tag(angle_deg: float) -> str:
    a = angle_deg % 360.0
    for lo, hi, tag in _NUANCED_TAGS:
        # "Uplifting" band wraps around 360°.
        if lo > hi:
            if a >= lo or a < hi:
                return tag
        else:
            if lo <= a < hi:
                return tag
    return "Uplifting"  # should be unreachable, safe fallback


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_mood(features_meta: Dict[str, Any]) -> MoodPoint:
    """
    Derive continuous valence and arousal from the aggregated audio features
    already produced by extract_features().

    Arguments
    ---------
    features_meta : dict
        The `.meta` attribute of a FeatureOutput. Must contain the keys
        `mfcc_mean`, `mfcc_std`, `tempo`, `rms_mean`,
        `spectral_centroid_mean`.

    Returns
    -------
    MoodPoint
        Dimensional mood coordinates and interpretable labels.

    Notes
    -----
    The weighting scheme follows the music-information-retrieval literature:

    * Arousal correlates most strongly with tempo, loudness/RMS, and
      spectral flux / high-frequency content (Soleymani et al. 2013; Yang
      & Chen 2012).
    * Valence is the harder dimension without chromatic / modal features;
      spectral centroid and high-MFCC brightness serve as approximations of
      perceived positivity, with a small positive contribution from tempo
      and a negative one from chaotic spectral variability.
    """
    mfcc_mean = np.asarray(features_meta.get("mfcc_mean", []), dtype=float)
    mfcc_std  = np.asarray(features_meta.get("mfcc_std",  []), dtype=float)

    if mfcc_mean.size == 0 or mfcc_std.size == 0:
        raise ValueError("features_meta must include non-empty 'mfcc_mean' and 'mfcc_std'.")

    # ── Acoustic proxies (identical normalisation ranges to emotion.py) ──
    variability = float(np.mean(mfcc_std))
    hi_band = float(np.mean(mfcc_mean[6:20])) if mfcc_mean.size >= 13 else float(np.mean(mfcc_mean))
    lo_band = float(np.mean(mfcc_mean[0:6]))  if mfcc_mean.size >= 6  else float(np.mean(mfcc_mean))
    tempo    = float(features_meta.get("tempo", 120.0))
    rms      = float(features_meta.get("rms_mean", 0.05))
    centroid = float(features_meta.get("spectral_centroid_mean", 2000.0))

    v_var    = _z(variability, 5.0,    80.0)
    v_bright = _z(hi_band,     -30.0,  150.0)
    v_warm   = _z(lo_band,     -350.0, 50.0)
    v_tempo  = _z(tempo,       60.0,   180.0)   # 60 BPM = calm, 180 = energetic
    v_energy = _z(rms,         0.0,    0.3)
    v_cent   = _z(centroid,    500.0,  6000.0)

    # ── Arousal: rhythmic & energetic activation ─────────────────────────
    arousal_01 = _clamp01(
        0.35 * v_tempo  +
        0.30 * v_energy +
        0.20 * v_var    +
        0.15 * v_cent
    )

    # ── Valence: brightness & tonal positivity ───────────────────────────
    valence_01 = _clamp01(
        0.35 * v_cent   +
        0.25 * v_bright +
        0.15 * v_warm   +
        0.15 * v_tempo  +
        0.10 * (1.0 - v_var)   # less chaotic = marginally more positive
    )

    # Rescale to [-1, 1] so the origin is "emotionally neutral".
    valence = 2.0 * valence_01 - 1.0
    arousal = 2.0 * arousal_01 - 1.0

    radius = math.sqrt(valence * valence + arousal * arousal)
    # Counter-clockwise from +valence axis, in degrees.
    angle_deg = (math.degrees(math.atan2(arousal, valence)) + 360.0) % 360.0

    quadrant    = _quadrant(valence, arousal)
    nuanced_tag = _nuanced_tag(angle_deg)
    confidence  = _clamp01(radius / math.sqrt(2.0))

    rationale = (
        f"Arousal {arousal:+.2f} (tempo {tempo:.0f} BPM, RMS {rms:.3f}); "
        f"valence {valence:+.2f} (centroid {centroid:.0f} Hz, brightness {v_bright:.2f}). "
        f"Places the track in the '{quadrant}' quadrant at {angle_deg:.0f}° "
        f"on the mood wheel, best described as {nuanced_tag}."
    )

    return MoodPoint(
        valence=round(valence, 4),
        arousal=round(arousal, 4),
        radius=round(radius, 4),
        angle_deg=round(angle_deg, 2),
        quadrant=quadrant,
        nuanced_tag=nuanced_tag,
        confidence=round(confidence, 4),
        rationale=rationale,
    )


def mood_from_discrete(scores: Dict[str, float]) -> MoodPoint:
    """
    Fallback mapping used for already-indexed tracks that only have the
    discrete 4-class scores (calm / energetic / happy / sad) stored.

    Provides approximate valence/arousal coordinates so the circumplex UI
    still renders for legacy index rows without re-extracting audio features.
    """
    calm      = float(scores.get("calm", 0.0))
    energetic = float(scores.get("energetic", 0.0))
    happy     = float(scores.get("happy", 0.0))
    sad       = float(scores.get("sad", 0.0))

    # Map each discrete emotion to a canonical unit-circle coordinate
    # (Russell's classic placement), then take the score-weighted average.
    anchors = {
        "energetic": ( 0.6,  0.8),   # Q1
        "happy":     ( 0.8,  0.3),   # Q1 (lower arousal than energetic)
        "calm":      ( 0.5, -0.7),   # Q4
        "sad":       (-0.7, -0.4),   # Q3
    }

    total = calm + energetic + happy + sad
    if total <= 0:
        valence, arousal = 0.0, 0.0
    else:
        valence = (
            happy     * anchors["happy"][0]    +
            energetic * anchors["energetic"][0] +
            calm      * anchors["calm"][0]     +
            sad       * anchors["sad"][0]
        ) / total
        arousal = (
            happy     * anchors["happy"][1]    +
            energetic * anchors["energetic"][1] +
            calm      * anchors["calm"][1]     +
            sad       * anchors["sad"][1]
        ) / total

    valence = _clamp(valence, -1.0, 1.0)
    arousal = _clamp(arousal, -1.0, 1.0)

    radius = math.sqrt(valence * valence + arousal * arousal)
    angle_deg = (math.degrees(math.atan2(arousal, valence)) + 360.0) % 360.0

    quadrant    = _quadrant(valence, arousal)
    nuanced_tag = _nuanced_tag(angle_deg)
    confidence  = _clamp01(radius / math.sqrt(2.0))

    rationale = (
        f"Approximated from discrete scores — "
        f"calm {calm:.2f}, energetic {energetic:.2f}, "
        f"happy {happy:.2f}, sad {sad:.2f}."
    )

    return MoodPoint(
        valence=round(valence, 4),
        arousal=round(arousal, 4),
        radius=round(radius, 4),
        angle_deg=round(angle_deg, 2),
        quadrant=quadrant,
        nuanced_tag=nuanced_tag,
        confidence=round(confidence, 4),
        rationale=rationale,
    )
