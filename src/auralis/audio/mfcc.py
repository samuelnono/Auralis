import librosa
import numpy as np

def extract_mfcc(path, n_mfcc=13):
    """
    Load an audio file and extract MFCC features.

    Args:
        path (str): Path to audio file
        n_mfcc (int): Number of MFCC coefficients

    Returns:
        dict: MFCC features and metadata
    """
    y, sr = librosa.load(path, sr=None)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    return {
        "mfcc": mfcc,
        "mfcc_mean": np.mean(mfcc, axis=1),
        "mfcc_std": np.std(mfcc, axis=1),
        "sr": sr,
        "duration_sec": len(y) / sr
    }
