# Data Directory

This directory contains data used for local development and experimentation.

## raw/
The `raw/` folder holds unprocessed audio files used for testing feature extraction
and emotion analysis pipelines.

- Files in `raw/` are intentionally excluded from version control.
- Users should place their own local audio samples here.

### Sample file (local only)
- `sample_01.wav`  
  A short mono WAV audio clip used to verify:
  - audio loading
  - spectral feature extraction
  - emotion mapping pipeline integration

No raw audio files are committed to the repository.
