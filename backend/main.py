"""
Auralis - FastAPI Backend
--------------------------
Exposes all Auralis modules via a REST API for the React frontend.
"""

import os
from dotenv import load_dotenv
load_dotenv()
import json
import tempfile
import requests
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.auralis.audio.features import extract_features
from src.auralis.audio.similarity import compare_features
from src.auralis.emotion.emotion import map_emotion
from src.auralis.preference.profile import UserProfile
from src.auralis.preference.feedback import record_feedback, feedback_summary
from src.auralis.preference.recommender import rank_songs, load_index
from src.auralis.playlist.generator import generate_playlist, playlist_to_csv
from src.auralis.chat.conversation import build_system_prompt, format_history_for_api

app = FastAPI(title="Auralis API")

# Allow React dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROFILE_PATH = "data/processed/user_profile.json"
INDEX_PATH   = "data/processed/research_index.csv"
MODEL        = "claude-haiku-4-5-20251001"


# ── Pydantic models ───────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    path: str
    label: str          # "like" | "dislike"
    emotion_label: str
    emotion_scores: dict
    vector: list

class RecommendRequest(BaseModel):
    alpha: float = 0.7
    top_k: int = 10
    exclude_rated: bool = True

class PlaylistRequest(BaseModel):
    mode: str = "emotion"           # "emotion" | "profile"
    target_emotion: Optional[str] = "calm"
    length: int = 8

class ChatRequest(BaseModel):
    messages: list
    last_track_meta: Optional[dict] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_profile() -> UserProfile:
    return UserProfile.load_or_new(PROFILE_PATH)

def get_index():
    try:
        return load_index(INDEX_PATH)
    except FileNotFoundError:
        return []


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Extract MFCC features and emotion from an uploaded audio file."""
    suffix = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        features = extract_features(tmp_path)
        emotion  = map_emotion(features.meta)
        return {
            "path":     tmp_path,
            "emotion":  emotion.emotion,
            "scores":   emotion.scores,
            "meta":     {k: v for k, v in features.meta.items() if k != "mfcc_mean" and k != "mfcc_std"},
            "vector":   features.vector.tolist(),
            "mfcc_mean": features.meta.get("mfcc_mean", []),
            "mfcc_std":  features.meta.get("mfcc_std", []),
            "tempo":     features.meta.get("tempo"),
            "rms_mean":  features.meta.get("rms_mean"),
            "spectral_centroid_mean": features.meta.get("spectral_centroid_mean"),
            "duration_sec": features.meta.get("duration_sec"),
            "sr":        features.meta.get("sr"),
            "vector_dim": features.meta.get("vector_dim"),
        }
    finally:
        os.unlink(tmp_path)


@app.post("/similarity")
async def similarity(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """Compare two audio files and return cosine similarity."""
    results = []
    paths = []
    for f in [file1, file2]:
        suffix = Path(f.filename).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await f.read())
            paths.append(tmp.name)
    try:
        f1 = extract_features(paths[0])
        f2 = extract_features(paths[1])
        from src.auralis.audio.similarity import cosine_similarity
        sim = cosine_similarity(f1.vector, f2.vector)
        return {"similarity": round(sim, 4)}
    finally:
        for p in paths:
            os.unlink(p)


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Record a like or dislike for a track."""
    import numpy as np
    profile = get_profile()
    profile = record_feedback(
        profile=profile,
        vector=np.array(req.vector),
        emotion_scores=req.emotion_scores,
        emotion_label=req.emotion_label,
        track_path=req.path,
        label=req.label,
        profile_save_path=PROFILE_PATH,
    )
    return feedback_summary(profile)


@app.get("/profile")
def profile():
    """Get the current user preference profile."""
    p = get_profile()
    return {
        "has_signal": p.has_signal(),
        "dominant_emotion": p.dominant_emotion(),
        "total_likes": p.total_likes,
        "total_dislikes": p.total_dislikes,
        "emotion_affinity": p.emotion_affinity,
        "interaction_log": p.interaction_log,
        "rated_paths": [e["track"] for e in p.interaction_log],
    }


@app.delete("/profile")
def reset_profile():
    """Reset the user profile."""
    UserProfile().save(PROFILE_PATH)
    return {"status": "reset"}


@app.post("/recommendations")
def recommendations(req: RecommendRequest):
    """Get ranked song recommendations based on user profile."""
    profile = get_profile()
    if not profile.has_signal():
        raise HTTPException(status_code=400, detail="No profile signal yet. Like some tracks first.")
    index = get_index()
    if not index:
        raise HTTPException(status_code=404, detail="Song index not found.")
    exclude = [e["track"] for e in profile.interaction_log] if req.exclude_rated else []
    recs = rank_songs(profile, index, alpha=req.alpha, exclude_paths=exclude, top_k=req.top_k)
    return {"recommendations": recs}


@app.post("/playlist")
def playlist(req: PlaylistRequest):
    """Generate a playlist by emotion or profile."""
    profile = get_profile()
    index = get_index()
    if not index:
        raise HTTPException(status_code=404, detail="Song index not found.")
    try:
        pl = generate_playlist(
            index=index,
            mode=req.mode,
            target_emotion=req.target_emotion,
            profile=profile if req.mode == "profile" else None,
            length=req.length,
        )
        return {"playlist": pl}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/playlist/export")
def playlist_export(mode: str = "emotion", target_emotion: str = "calm", length: int = 8):
    """Export a playlist as CSV."""
    from fastapi.responses import Response
    profile = get_profile()
    index = get_index()
    pl = generate_playlist(index=index, mode=mode, target_emotion=target_emotion,
                           profile=profile, length=length)
    csv_data = playlist_to_csv(pl)
    return Response(content=csv_data, media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=auralis_playlist.csv"})


@app.get("/index")
def index_songs():
    """Return the full song index."""
    index = get_index()
    songs = []
    for row in index:
        songs.append({
            "path": row.get("path", ""),
            "name": Path(row.get("path", "unknown")).stem,
            "emotion": row.get("predicted_emotion") or row.get("emotion", "unknown"),
            "calm":     float(row.get("calm", 0)),
            "energetic":float(row.get("energetic", 0)),
            "happy":    float(row.get("happy", 0)),
            "sad":      float(row.get("sad", 0)),
        })
    return {"songs": songs, "total": len(songs)}


@app.post("/chat")
def chat(req: ChatRequest):
    """Send a message to the Auralis LLM assistant."""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    profile = get_profile()
    index = get_index()

    system = build_system_prompt(
        profile=profile,
        index=index,
        last_analyzed_track=req.last_track_meta,
    )

    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "system": system,
        "messages": req.messages,
    }

    #  Try real API first
    if api_key:
        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                text = "".join(
                    b.get("text", "")
                    for b in data.get("content", [])
                    if b.get("type") == "text"
                )
                return {"response": text}

            else:
                print("⚠️ Anthropic failed, switching to fallback mode")
                print(response.text)

        except Exception as e:
            print("⚠️ Exception calling Anthropic:", e)

    #  FALLBACK MODE (DEMO SAFE)
    last_user_msg = req.messages[-1]["content"] if req.messages else ""

    fallback_response = f"""
Auralis Insight:

Based on your question: "{last_user_msg}"

This system analyzes music through frequency structure using MFCC features and maps them into emotional space.

At this stage, recommendations are driven by:
- Acoustic similarity (cosine similarity)
- Emotion alignment (calm, energetic, happy, sad)

Future versions will incorporate:
- User preference learning
- Playlist generation
- Conversational intelligence

(This is a fallback response due to API limitations.)
"""

    return {"response": fallback_response.strip()}