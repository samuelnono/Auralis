import csv
import json
from pathlib import Path

from src.auralis.audio.features import extract_features
from src.auralis.emotion.emotion import map_emotion

INPUT_DIR   = Path("data/raw/songs")
OUTPUT_FILE = Path("data/processed/research_index.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

FIELDNAMES = [
    "path",
    "predicted_emotion",
    "calm",
    "energetic",
    "happy",
    "sad",
    "vector_dim",
    "mfcc_vector",   # ← NEW: JSON-serialised 26-dim feature vector
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

            row = {
                "path":             str(file_path),
                "predicted_emotion": emotion.emotion,
                "calm":             emotion.scores["calm"],
                "energetic":        emotion.scores["energetic"],
                "happy":            emotion.scores["happy"],
                "sad":              emotion.scores["sad"],
                "vector_dim":       features.meta["vector_dim"],
                "mfcc_vector":      json.dumps(features.vector.tolist()),
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
