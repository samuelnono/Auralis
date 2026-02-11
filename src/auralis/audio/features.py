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
    n_mfcc: int = 13,
    include: Iterable[str] = ("mfcc_mean", "mfcc_std"),
) -> FeatureOutput:

    # Call MFCC extractor (matching your mfcc.py signature)
    mfcc_out = extract_mfcc(path, n_mfcc=n_mfcc)

    mfcc = np.asarray(mfcc_out["mfcc"], dtype=float)

    # Compute statistics
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    if mfcc_mean.size == 0 or mfcc_std.size == 0:
        raise ValueError("MFCC mean/std came back empty.")

    parts: List[np.ndarray] = []
    features_used: List[str] = []

    for name in include:
        if name == "mfcc_mean":
            parts.append(mfcc_mean)
            features_used.append("mfcc_mean")
        elif name == "mfcc_std":
            parts.append(mfcc_std)
            features_used.append("mfcc_std")
        else:
            raise ValueError(f"Unknown feature '{name}'")

    vector = np.concatenate(parts).astype(float)

    meta = {
        "path": path,
        "sr": mfcc_out.get("sr"),
        "duration_sec": mfcc_out.get("duration_sec"),
        "n_mfcc": n_mfcc,
        "mfcc_shape": tuple(mfcc.shape),
        "vector_dim": vector.shape[0],
        "features": features_used,
        "mfcc_mean": mfcc_mean.tolist(),
        "mfcc_std": mfcc_std.tolist(),
    }

    return FeatureOutput(vector=vector, meta=meta)


