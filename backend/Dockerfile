FROM python:3.11-slim

WORKDIR /app

# System dependencies for librosa / audio decoding.
# libsndfile1 is required by soundfile; ffmpeg lets librosa read mp3.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY backend ./backend
COPY src ./src
COPY data ./data

EXPOSE 8000

# Two workers is a sensible default for a small Fly.io VM; tune up on larger
# instances. Fly routes traffic to port 8000 via the internal_port in fly.toml.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
