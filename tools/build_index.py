import os
import csv
from pathlib import Path

from src.auralis.audio.features import extract_features
from src.auralis.emotion.emotion import map_emotion

INPUT_DIR = Path("data/raw/songs")
OUTPUT_FILE = Path("data/processed/research_index.csv")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

rows = []

print("Indexing songs...\n")

for file_path in INPUT_DIR.glob("*.wav"):
    print(f"Processing: {file_path.name}")

    features = extract_features(str(file_path))
    emotion = map_emotion(features.meta)

    row = {
        "path": str(file_path),
        "predicted_emotion": emotion.emotion,
        "calm": emotion.scores["calm"],
        "energetic": emotion.scores["energetic"],
        "happy": emotion.scores["happy"],
        "sad": emotion.scores["sad"],
        "vector_dim": features.meta["vector_dim"]
    }

    rows.append(row)

with open(OUTPUT_FILE, "w", newline="") as csvfile:
    fieldnames = rows[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nIndex created: {OUTPUT_FILE}")
print(f"Songs indexed: {len(rows)}")