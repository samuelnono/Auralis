import librosa
import numpy as np


def extract_mfcc(path: str, n_mfcc: int = 20) -> dict:
    """
    Load an audio file and extract MFCC features plus supplementary
    acoustic descriptors for richer similarity comparison.

    Args:
        path   : Path to audio file (.wav or .mp3)
        n_mfcc : Number of MFCC coefficients (default raised to 20)

    Returns:
        dict with keys:
          mfcc          – raw MFCC matrix (n_mfcc x frames)
          mfcc_mean     – per-coefficient mean  (n_mfcc,)
          mfcc_std      – per-coefficient std   (n_mfcc,)
          tempo         – estimated BPM (float)
          rms_mean      – mean RMS energy (float)
          rms_std       – std  RMS energy (float)
          spectral_centroid_mean – mean spectral centroid in Hz (float)
          spectral_centroid_std  – std  spectral centroid in Hz (float)
          sr            – sample rate (int)
          duration_sec  – track duration in seconds (float)
    """
    y, sr = librosa.load(path, sr=None)

    # ── MFCCs (expanded to 20 coefficients) ─────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # ── Tempo ────────────────────────────────────────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # ── RMS energy ───────────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]

    # ── Spectral centroid ────────────────────────────────────────────────────
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    return {
        "mfcc":                    mfcc,
        "mfcc_mean":               np.mean(mfcc, axis=1),
        "mfcc_std":                np.std(mfcc, axis=1),
        "tempo":                   tempo,
        "rms_mean":                float(np.mean(rms)),
        "rms_std":                 float(np.std(rms)),
        "spectral_centroid_mean":  float(np.mean(spec_centroid)),
        "spectral_centroid_std":   float(np.std(spec_centroid)),
        "sr":                      sr,
        "duration_sec":            len(y) / sr,
    }
