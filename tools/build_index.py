"""
Auralis — Research Index Builder (v1.1)
----------------------------------------
Walks `data/raw/songs/`, extracts acoustic features, computes both the
discrete 4-class emotion (calm / energetic / happy / sad) and the new
continuous valence-arousal mood coordinates, and writes everything to
`data/processed/research_index.csv`.

Columns (v1.1):
    path, predicted_emotion,
    calm, energetic, happy, sad,
    valence, arousal, radius, angle_deg, quadrant, nuanced_tag, confidence,
    vector_dim, mfcc_vector (JSON-serialised feature vector)

Legacy rows written by earlier versions are still readable by the backend —
the /index endpoint synthesises approximate mood coordinates from the
discrete scores when the continuous columns are missing.

Run from the project root:
    python -m tools.build_index
"""

import csv
import json
from pathlib import Path

from src.auralis.audio.features import extract_features
from src.auralis.emotion.emotion import map_emotion
from src.auralis.emotion.valence_arousal import compute_mood

INPUT_DIR   = Path("data/raw/songs")
OUTPUT_FILE = Path("data/processed/research_index.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "path",
    "predicted_emotion",
    # Discrete 4-class baseline
    "calm", "energetic", "happy", "sad",
    # Continuous valence-arousal layer (v1.1)
    "valence", "arousal", "radius", "angle_deg",
    "quadrant", "nuanced_tag", "confidence",
    # Feature vector metadata
    "vector_dim",
    "mfcc_vector",
]

rows = []
print("Indexing songs…\n")

# Support both .wav and .mp3
audio_files = sorted(
    list(INPUT_DIR.glob("*.wav")) + list(INPUT_DIR.glob("*.mp3"))
)

if not audio_files:
    print(f"No audio files found in {INPUT_DIR}. Nothing to index.")
else:
    for file_path in audio_files:
        print(f"Processing: {file_path.name}")
        try:
            features = extract_features(str(file_path))
            emotion  = map_emotion(features.meta)
            mood     = compute_mood(features.meta)

            row = {
                "path":              str(file_path),
                "predicted_emotion": emotion.emotion,
                "calm":              emotion.scores["calm"],
                "energetic":         emotion.scores["energetic"],
                "happy":             emotion.scores["happy"],
                "sad":               emotion.scores["sad"],
                "valence":           mood.valence,
                "arousal":           mood.arousal,
                "radius":            mood.radius,
                "angle_deg":         mood.angle_deg,
                "quadrant":          mood.quadrant,
                "nuanced_tag":       mood.nuanced_tag,
                "confidence":        mood.confidence,
                "vector_dim":        features.meta["vector_dim"],
                "mfcc_vector":       json.dumps(features.vector.tolist()),
            }
            rows.append(row)

        except Exception as e:
            print(f"    Skipped {file_path.name}: {e}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n Index saved:    {OUTPUT_FILE}")
    print(f"   Songs indexed:  {len(rows)}")
    print(f"   Songs skipped:  {len(audio_files) - len(rows)}")
