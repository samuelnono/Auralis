import numpy as np


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors.
    """
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Inputs must be 1D vectors")

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def compare_features(feat_a, feat_b) -> dict:
    """
    Compare two FeatureOutput objects produced by extract_features().
    """
    vec_a = feat_a.vector
    vec_b = feat_b.vector

    score = cosine_similarity(vec_a, vec_b)

    return {
        "similarity": round(score, 4),
        "path_a": feat_a.meta["path"],
        "path_b": feat_b.meta["path"]
    }
