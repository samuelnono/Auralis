# src/auralis/audio/features.py
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List
import numpy as np

from .mfcc import extract_mfcc


@dataclass(frozen=True)
class FeatureOutput:
    vector: np.ndarray
    meta: Dict[str, Any]


def extract_features(
    path: str,
    n_mfcc: int = 20,
    include: Iterable[str] = (
        "mfcc_mean",
        "mfcc_std",
        "tempo",
        "rms_mean",
        "rms_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
    ),
) -> FeatureOutput:
    """
    Extract a fixed-length feature vector from an audio file.

    Default vector composition (46 dimensions):
      - 20 MFCC means
      - 20 MFCC stds
      - 1  tempo (BPM, normalised)
      - 1  RMS energy mean
      - 1  RMS energy std
      - 1  spectral centroid mean (normalised)
      - 1  spectral centroid std  (normalised)

    All scalar features are normalised to a comparable range so they
    don't dominate the cosine similarity computation.
    """
    mfcc_out = extract_mfcc(path, n_mfcc=n_mfcc)
    mfcc = np.asarray(mfcc_out["mfcc"], dtype=float)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std  = np.std(mfcc,  axis=1)

    if mfcc_mean.size == 0 or mfcc_std.size == 0:
        raise ValueError("MFCC mean/std came back empty.")

    # Normalise scalar features to roughly [0, 1] so they sit in the
    # same numeric range as MFCC coefficients and don't skew cosine sim.
    tempo_norm     = mfcc_out["tempo"] / 200.0          # typical BPM 60-180
    rms_mean_norm  = mfcc_out["rms_mean"] / 0.5         # typical RMS 0-0.5
    rms_std_norm   = mfcc_out["rms_std"]  / 0.2
    sc_mean_norm   = mfcc_out["spectral_centroid_mean"] / 8000.0   # Hz
    sc_std_norm    = mfcc_out["spectral_centroid_std"]  / 4000.0

    scalar_map = {
        "tempo":                   np.array([tempo_norm]),
        "rms_mean":                np.array([rms_mean_norm]),
        "rms_std":                 np.array([rms_std_norm]),
        "spectral_centroid_mean":  np.array([sc_mean_norm]),
        "spectral_centroid_std":   np.array([sc_std_norm]),
    }

    parts: List[np.ndarray] = []
    features_used: List[str] = []

    for name in include:
        if name == "mfcc_mean":
            parts.append(mfcc_mean)
            features_used.append("mfcc_mean")
        elif name == "mfcc_std":
            parts.append(mfcc_std)
            features_used.append("mfcc_std")
        elif name in scalar_map:
            parts.append(scalar_map[name])
            features_used.append(name)
        else:
            raise ValueError(f"Unknown feature '{name}'")

    vector = np.concatenate(parts).astype(float)

    meta = {
        "path":                    path,
        "sr":                      mfcc_out.get("sr"),
        "duration_sec":            mfcc_out.get("duration_sec"),
        "n_mfcc":                  n_mfcc,
        "mfcc_shape":              tuple(mfcc.shape),
        "vector_dim":              vector.shape[0],
        "features":                features_used,
        # Raw values for emotion mapping & display
        "mfcc_mean":               mfcc_mean.tolist(),
        "mfcc_std":                mfcc_std.tolist(),
        "tempo":                   mfcc_out["tempo"],
        "rms_mean":                mfcc_out["rms_mean"],
        "rms_std":                 mfcc_out["rms_std"],
        "spectral_centroid_mean":  mfcc_out["spectral_centroid_mean"],
        "spectral_centroid_std":   mfcc_out["spectral_centroid_std"],
    }

    return FeatureOutput(vector=vector, meta=meta)
