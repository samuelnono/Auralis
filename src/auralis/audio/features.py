"""
src/auralis/audio/features.py

Phase 2B — Feature Aggregator

Goal:
- One function that calls MFCC extraction (and later other features),
  aggregates them into a single fixed-length vector, and returns metadata.

Current features:
- MFCC mean (n_mfcc,)
- MFCC std  (n_mfcc,)

Total dimension: 2 * n_mfcc
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import numpy as np

from src.auralis.audio.mfcc import extract_mfcc


@dataclass(frozen=True)
class FeatureOutput:
    """Returned by extract_features()."""
    vector: np.ndarray            # shape: (D,)
    meta: Dict[str, Any]          # extra info for debugging/research tracking


def extract_features(
    path: str,
    *,
    n_mfcc: int = 13,
) -> FeatureOutput:
    """
    Extract a single fixed-length feature vector from an audio file.

    Parameters
    ----------
    path : str
        Path to audio file.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    FeatureOutput
        .vector: np.ndarray (D,)
        .meta: dict
    """
    mfcc_out = extract_mfcc(path, n_mfcc=n_mfcc)

    # mfcc_out is a dict produced by mfcc.py
    # Expected keys: mfcc, mfcc_mean, mfcc_std, sr, duration_sec
    mfcc = np.asarray(mfcc_out.get("mfcc"))
    if mfcc.ndim != 2:
        raise ValueError(
            f"Expected mfcc to be 2D (n_mfcc, T). Got shape {mfcc.shape} (ndim={mfcc.ndim})."
        )

    # Prefer the precomputed stats from mfcc.py if present
    mfcc_mean = mfcc_out.get("mfcc_mean", np.mean(mfcc, axis=1))
    mfcc_std = mfcc_out.get("mfcc_std", np.std(mfcc, axis=1))

    mfcc_mean = np.asarray(mfcc_mean).reshape(-1)
    mfcc_std = np.asarray(mfcc_std).reshape(-1)

    vector = np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(np.float32)

    meta: Dict[str, Any] = {
        "path": path,
        "sr": mfcc_out.get("sr"),
        "duration_sec": mfcc_out.get("duration_sec"),
        "n_mfcc": int(n_mfcc),
        "mfcc_shape": tuple(mfcc.shape),
        "vector_dim": int(vector.shape[0]),
        "features": ["mfcc_mean", "mfcc_std"],
    }

    return FeatureOutput(vector=vector, meta=meta)
