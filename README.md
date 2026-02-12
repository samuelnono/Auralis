Roadmap

Setup ✅
Git + GitHub repository

Python virtual environment

Dependency management (requirements.txt)


Phase 1 — Core Audio Pipeline ✅

Audio loading (WAV / MP3 supported)

MFCC feature extraction

Feature aggregation (mean + standard deviation)

Cosine similarity comparison


Phase 2 — Emotion Mapping (Interpretable) ✅

Rule-based emotion inference from MFCC statistics

Multi-emotion scoring (calm, energetic, happy, sad)

Transparent score output


Phase 3 — Interactive Frontend ✅

Streamlit web interface

Upload one or two audio files

Emotion prediction display

Similarity analysis

CSV experiment logging


Phase 4 — Ongoing Development ⏳

Expanded spectral features

Improved emotion heuristics

Scalable recommendation engine

Research evaluation and publication



 Current System Capabilities

✔ Audio feature extraction (MFCC-based)
✔ Statistical feature aggregation
✔ Interpretable emotion prediction
✔ Cosine similarity comparison
✔ Experiment logging (CSV export)
✔ End-to-end Streamlit demo



 System Architecture

The system follows a modular, research-oriented pipeline.

User Upload (WAV / MP3)
        │
        ▼
Audio Loader (librosa)
        │
        ▼
MFCC Extraction
        │
        ▼
Feature Aggregation
(mean + std statistics)
        │
        ├──────────────► Cosine Similarity Module
        │
        ▼
Emotion Mapping Engine
(rule-based, interpretable)
        │
        ▼
Streamlit Frontend Output
- Emotion scores
- Predicted emotion
- Similarity score
- CSV logging


Each component is isolated and replaceable.
The feature extraction layer can later be swapped for deep embeddings.
The emotion mapping module can be replaced with a learned classifier.
The similarity engine can scale to large music databases.

The design is intentional: clarity first, complexity later.
