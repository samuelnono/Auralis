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
from collections import defaultdict
from time import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.auralis.audio.features import extract_features
from src.auralis.audio.similarity import compare_features
from src.auralis.emotion.emotion import map_emotion
from src.auralis.emotion.valence_arousal import compute_mood, mood_from_discrete
from src.auralis.preference.profile import UserProfile
from src.auralis.preference.feedback import (
    record_feedback,
    record_spotify_feedback,
    feedback_summary,
)
from src.auralis.preference.recommender import rank_songs, load_index
from src.auralis.playlist.generator import generate_playlist, playlist_to_csv
from src.auralis.chat.conversation import build_system_prompt, format_history_for_api
from src.auralis.spotify import SpotifyClient, SpotifyError, mood_to_query

app = FastAPI(title="Auralis API")

# CORS origins are env-driven so the same image works in dev, staging and prod.
# Locally, the defaults cover the Vite (5173) and CRA (3000) dev servers under
# both `localhost` and `127.0.0.1` — browsers treat these as distinct origins,
# and the Vite dev server is pinned to 127.0.0.1 in vite.config.js so cookies
# and Spotify's PKCE redirect URI line up. We allow both spellings so it
# doesn't matter which the user types into their address bar.
# In production (Fly.io), set CORS_ORIGINS to the Vercel URL, e.g.:
#     flyctl secrets set CORS_ORIGINS=https://auralis.vercel.app
_DEFAULT_CORS_ORIGINS = ",".join([
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
])
CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROFILE_PATH = "data/processed/user_profile.json"
INDEX_PATH   = "data/processed/research_index.csv"
MODEL        = "claude-haiku-4-5-20251001"

# ── Per-IP rate limiting for /chat (in-memory, process-local) ─────────────────
# Keeps the public demo cheap: each IP gets N chat messages per rolling hour.
# Tune via env vars; on Fly.io set e.g. `flyctl secrets set CHAT_RATE_LIMIT=10`.
CHAT_RATE_LIMIT      = int(os.environ.get("CHAT_RATE_LIMIT", "10"))
CHAT_RATE_WINDOW_SEC = int(os.environ.get("CHAT_RATE_WINDOW_SEC", "3600"))
_chat_hits: dict = defaultdict(list)

def _client_ip(request: Request) -> str:
    """Best-effort real client IP behind reverse proxies (Fly.io, Vercel)."""
    for header in ("fly-client-ip", "x-forwarded-for", "x-real-ip"):
        val = request.headers.get(header)
        if val:
            return val.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

def _enforce_chat_rate(request: Request) -> None:
    ip = _client_ip(request)
    now = time()
    window_start = now - CHAT_RATE_WINDOW_SEC
    hits = [t for t in _chat_hits[ip] if t > window_start]
    if len(hits) >= CHAT_RATE_LIMIT:
        retry_in = int(hits[0] + CHAT_RATE_WINDOW_SEC - now)
        raise HTTPException(
            status_code=429,
            detail=(
                f"Chat rate limit reached ({CHAT_RATE_LIMIT}/hour). "
                f"Try again in about {max(retry_in, 1) // 60 + 1} min."
            ),
        )
    hits.append(now)
    _chat_hits[ip] = hits


# ── Pydantic models ───────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    path: str
    label: str          # "like" | "dislike"
    emotion_label: str
    emotion_scores: dict
    vector: list
    # Original filename from the user's upload, preserved so the Interaction
    # History can show "Airplane Mode - Tilden Parc.wav" instead of the opaque
    # tmpXXXXXX path Python's tempfile module hands us on the analyze call.
    display_name: Optional[str] = None

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
    # Optional Spotify context the frontend can pass when the user is signed
    # in via the PKCE flow. The user's access token never reaches us — instead
    # the browser pre-fetches /me/top/tracks and /me/top/artists and forwards
    # the de-personalised highlights here so the LLM can ground taste-related
    # answers in real listening data.
    #
    # Expected shape:
    #   {"top_artists": ["Dave", "Baby Keem", ...],
    #    "top_tracks":  [{"name": "...", "artists": ["..."]}, ...],
    #    "display_name": "Samuel"}
    spotify_context: Optional[dict] = None

class SpotifyFeedbackRequest(BaseModel):
    """Heart/dislike on a Spotify card (For You rail, Playlist results, etc.)."""

    spotify_id:       str
    track_name:       str
    artists:          list = []
    label:            str = "like"   # "like" | "dislike"
    # The mood the user was browsing when they liked the track. Drives the
    # synthetic emotion scores fed into the affinity centroid; falls back to
    # ``calm`` if nothing is supplied so the affinity update is always defined.
    discrete_emotion: Optional[str] = None
    valence:          Optional[float] = None
    arousal:          Optional[float] = None


class SpotifyMoodRequest(BaseModel):
    valence: Optional[float] = None
    arousal: Optional[float] = None
    discrete_emotion: Optional[str] = None
    limit: int = 10
    # Optional escape hatch — the frontend can build its own Search query
    # (e.g. an artist:"..." OR artist:"..." clause derived from the user's
    # Spotify top artists) and have it dispatched through the same
    # backend-credentialed search path. When set, the mood inputs above are
    # ignored except for the rationale string.
    query_override: Optional[str] = None
    # Seeds from the logged-in user's Spotify history. When supplied, the
    # backend mixes them into the search-query pool so the playlist blends
    # the requested mood with the user's actual listening taste rather than
    # only returning generic mood-keyword results.
    #
    # ``seed_artists``: top artist names (strings).
    # ``seed_genres``:  optional genre tags pulled from those top artists.
    seed_artists: Optional[list] = None
    seed_genres:  Optional[list] = None
    # Diversification controls.
    #
    # ``exclude_ids``: spotify track IDs the user has already hearted in
    # Auralis (or seen recently in this session). Filtered out of the
    # search results so successive Generate presses surface new music
    # instead of recycling the same top tracks.
    #
    # ``shuffle_seed``: optional integer salt the frontend can supply to
    # vary the offset/query traversal order. When omitted, the backend
    # rolls a fresh per-call random seed so the lineup changes between
    # presses even for the same mood + seeds.
    exclude_ids:   Optional[list] = None
    shuffle_seed:  Optional[int]  = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_profile() -> UserProfile:
    return UserProfile.load_or_new(PROFILE_PATH)

def get_index():
    try:
        return load_index(INDEX_PATH)
    except FileNotFoundError:
        return []


# Lazy singleton so we only attempt the Client Credentials flow once we
# actually have to. Lets the rest of the API boot even if Spotify creds
# are missing (e.g. in a CI environment).
_spotify_client: Optional[SpotifyClient] = None

def get_spotify_client() -> SpotifyClient:
    global _spotify_client
    if _spotify_client is None:
        _spotify_client = SpotifyClient()
    return _spotify_client


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
        # Trim to a 30s mono sample at 22kHz so memory stays bounded on
        # small Fly.io VMs. Emotion is stable across a track; this gives
        # the same recommendation signal at ~5% of the memory cost.
        features = extract_features(tmp_path, max_duration=30.0, target_sr=22050)
        emotion  = map_emotion(features.meta)
        mood     = compute_mood(features.meta)
        return {
            "path":     tmp_path,
            "emotion":  emotion.emotion,
            "scores":   emotion.scores,
            "mood":     mood.as_dict(),
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
        f1 = extract_features(paths[0], max_duration=30.0, target_sr=22050)
        f2 = extract_features(paths[1], max_duration=30.0, target_sr=22050)
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
        display_name=req.display_name,
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


@app.get("/profile/spotify-likes")
def profile_spotify_likes():
    """Return the set of Spotify track IDs the user has liked in Auralis.

    Replaces the (now blocked) ``GET /me/tracks/contains`` Spotify call as
    the data source for heart-fill state on the Spotify rails. The frontend
    queries this on mount so previously-hearted tracks render as filled
    immediately, instead of resetting every time the user navigates away
    and back.
    """
    p = get_profile()
    liked: set[str] = set()
    disliked: set[str] = set()
    for entry in p.interaction_log:
        track = entry.get("track", "")
        if not track.startswith("spotify:track:"):
            continue
        sid = track.split(":", 2)[-1]
        if not sid:
            continue
        if entry.get("feedback") == "like":
            liked.add(sid)
        elif entry.get("feedback") == "dislike":
            disliked.add(sid)

    return {
        "liked":    sorted(liked),
        "disliked": sorted(disliked),
    }


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
    """Return the full song index, including dimensional mood coordinates.

    If the CSV was produced by the new (v1.1+) indexer it will contain
    valence/arousal columns directly. For legacy rows we synthesise
    approximate coordinates from the discrete 4-class scores so the
    circumplex UI still works without re-indexing.
    """
    index = get_index()
    songs = []
    for row in index:
        scores = {
            "calm":      float(row.get("calm", 0)      or 0),
            "energetic": float(row.get("energetic", 0) or 0),
            "happy":     float(row.get("happy", 0)     or 0),
            "sad":       float(row.get("sad", 0)       or 0),
        }

        # Prefer stored continuous coordinates when available.
        try:
            valence  = float(row["valence"])
            arousal  = float(row["arousal"])
            quadrant = row.get("quadrant") or ""
            nuanced  = row.get("nuanced_tag") or ""
            has_mood = True
        except (KeyError, TypeError, ValueError):
            has_mood = False

        if not has_mood:
            mp = mood_from_discrete(scores)
            valence  = mp.valence
            arousal  = mp.arousal
            quadrant = mp.quadrant
            nuanced  = mp.nuanced_tag

        songs.append({
            "path":    row.get("path", ""),
            "name":    Path(row.get("path", "unknown")).stem,
            "emotion": row.get("predicted_emotion") or row.get("emotion", "unknown"),
            **scores,
            "mood": {
                "valence":     valence,
                "arousal":     arousal,
                "quadrant":    quadrant,
                "nuanced_tag": nuanced,
            },
        })
    return {"songs": songs, "total": len(songs)}


@app.post("/chat")
def chat(req: ChatRequest, request: Request):
    """Send a message to the Auralis LLM assistant."""

    # Per-IP rolling-hour rate limit to keep the public demo affordable.
    _enforce_chat_rate(request)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()

    profile = get_profile()
    index = get_index()

    # ── Fresh Spotify candidate pool ─────────────────────────────────────────
    # Pre-fetch a *cross-mood* set of Spotify search results so the LLM has
    # real recommendations for whatever the user asks — not just their
    # dominant emotion. We pull ~5 tracks for each of {energetic, calm,
    # happy, sad}, blended with the user's Spotify top artists, and tag
    # each candidate with its mood so the LLM can pick the right slice
    # depending on the question. Already-liked tracks are excluded so
    # suggestions stay fresh.
    spotify_candidates: list[dict] = []
    detected_artist_genres: dict = {}  # surfaced into the system prompt
    try:
        # Top artist + genre seeds from the Spotify context the frontend
        # forwards. Genres are *critical* — without them, an "X-style"
        # request matches on mood keywords alone and pulls in tracks from
        # totally different genre families (e.g. country tracks for a UK
        # rap artist seed). With them, every search slot is biased toward
        # the listener's genre neighborhood.
        seed_artists: list[str] = []
        seed_genres:  list[str] = []
        artist_genre_map: dict = {}    # name(lower) -> [genres]
        artist_id_map:    dict = {}    # name(lower) -> spotify_id
        if req.spotify_context:
            seed_artists = [
                a for a in (req.spotify_context.get("top_artists") or [])
                if isinstance(a, str) and a.strip()
            ][:5]
            seed_genres = [
                g for g in (req.spotify_context.get("top_genres") or [])
                if isinstance(g, str) and g.strip()
            ][:4]
            for ag in (req.spotify_context.get("artist_genres") or []):
                if isinstance(ag, dict):
                    n = (ag.get("name") or "").strip()
                    gs = [g for g in (ag.get("genres") or []) if isinstance(g, str)]
                    sid = (ag.get("spotify_id") or "").strip()
                    if n:
                        artist_genre_map[n.lower()] = gs
                        if sid:
                            artist_id_map[n.lower()] = sid

        # Already-hearted Spotify IDs to exclude. Computed up-front so the
        # priority-artist top-tracks block (which fires before the mood
        # loop) can also filter against them.
        liked_ids: set[str] = set()
        for entry in profile.interaction_log:
            track = entry.get("track", "")
            if (
                track.startswith("spotify:track:")
                and entry.get("feedback") == "like"
            ):
                sid = track.split(":", 2)[-1]
                if sid:
                    liked_ids.add(sid)

        # ── Artist mentions across recent user turns ────────────────────────
        # Detect any artist the user named (their top artists, or any other
        # name introduced via "X-style", "X vibes", "X-coded", "like X").
        # For each detected artist NOT in the user's top list, do a quick
        # Spotify artist-search to fetch their genre tags. The result is a
        # narrow, artist-aligned genre pool that completely overrides the
        # generic top-genre seeds for this turn — so "Baby Keem coded"
        # actually pulls Pglang/Compton hip-hop instead of Memphis trap.
        #
        # We scan the last THREE user turns (latest first, weighted) so
        # follow-ups like "yes" / "more like that" inherit the artist
        # context from the prior request instead of falling back to
        # generic mood picks. Latest-turn mentions take priority over
        # older ones during dedup.
        recent_user_msgs: list[str] = []
        for m in reversed(req.messages or []):
            if isinstance(m, dict) and m.get("role") == "user":
                recent_user_msgs.append(str(m.get("content") or ""))
                if len(recent_user_msgs) >= 3:
                    break
        latest_user_msg = recent_user_msgs[0] if recent_user_msgs else ""
        latest_user_msg_lc = latest_user_msg.lower()
        # Concatenated context for the regex passes — newest turn is at
        # the start so artist names from the most recent message win
        # the dedupe-by-first-seen.
        recent_context_text = "\n".join(recent_user_msgs)
        recent_context_lc   = recent_context_text.lower()

        named_artists: list[str] = []

        # Normalised form for fuzzy substring matching. Strips dots,
        # apostrophes, and collapses whitespace so "J. Cole", "J.Cole",
        # "j cole", and "j.cole" all collide. Without this, a user
        # typing "j.cole" doesn't match the top-artist name "J. Cole"
        # and detection silently fails.
        import re as _re_norm
        def _norm(s: str) -> str:
            return _re_norm.sub(r"[\s\.\']+", "", (s or "").lower())

        recent_context_norm = _norm(recent_context_text)

        # 1) Any of the user's top artists mentioned by name in the
        # recent-turns window. Highest signal — these are artists we
        # already know fit them. Substring search runs in BOTH the
        # raw lowercased form (for normal mentions like "dave coded")
        # AND the normalised form (for messy inputs like "j.cole").
        for a in seed_artists:
            if not a:
                continue
            if a.lower() in recent_context_lc and a not in named_artists:
                named_artists.append(a)
                continue
            if _norm(a) and _norm(a) in recent_context_norm and a not in named_artists:
                named_artists.append(a)

        # 2) Trigger-anchored extraction. Find each trigger word ("vibes",
        # "style", "coded", "like", etc.) and look back 1-2 words to find
        # the artist name. Stopwords are stripped so "give me kanye west
        # vibes" yields "kanye west", not "give me kanye west" — which was
        # the bug producing Spandau Ballet recs for a Kanye prompt.
        import re as _re
        _STOPWORDS = {
            "i", "we", "you", "me", "my", "your", "give", "show", "play",
            "want", "need", "find", "get", "recommend", "search", "for",
            "some", "something", "the", "a", "an", "of", "more", "another",
            "any", "to", "and", "or", "with", "but", "thats", "that's",
            "really", "super", "very", "kind", "kinda", "type", "in",
            "from", "by", "on",
        }

        # Triggers SUFFIXED (text → ... TRIGGER): "kanye west vibes"
        suffix_triggers = (
            r"vibes?", r"vibe", r"coded", r"style", r"styled", r"esque",
            r"-?ish", r"sounding", r"inspired", r"feel", r"feels",
            r"flavou?red?", r"cuts?",
        )
        suffix_pat = _re.compile(
            r"\b((?:[a-z][\w'\.\-]*\s+){0,2}[a-z][\w'\.\-]*)\s+"
            r"(?:" + "|".join(suffix_triggers) + r")\b",
            _re.IGNORECASE,
        )

        # Triggers PREFIXED (TRIGGER → ... text): "like kanye west"
        prefix_triggers = (
            r"like", r"sounding\s+like", r"similar\s+to",
            r"in\s+the\s+style\s+of", r"style\s+of", r"reminiscent\s+of",
            r"akin\s+to", r"vibes?\s+of", r"vibes?\s+from", r"vibes?\s+like",
        )
        prefix_pat = _re.compile(
            r"\b(?:" + "|".join(prefix_triggers) + r")\s+"
            r"((?:[a-z][\w'\.\-]*\s+){0,2}[a-z][\w'\.\-]*)\b",
            _re.IGNORECASE,
        )

        def _strip_stopwords(phrase: str) -> str:
            words = [w for w in phrase.split() if w]
            # Drop leading stopwords; we want the artist name only.
            while words and words[0].lower() in _STOPWORDS:
                words.pop(0)
            # Also drop *trailing* stopwords (rare but possible).
            while words and words[-1].lower() in _STOPWORDS:
                words.pop()
            return " ".join(words).strip()

        def _add_candidate(phrase: str):
            cleaned = _strip_stopwords(phrase)
            if not cleaned:
                return
            # Title-case for the Spotify lookup display name.
            cand = " ".join(w.capitalize() for w in cleaned.split())
            if cand.lower() not in {x.lower() for x in named_artists}:
                named_artists.append(cand)

        for m in suffix_pat.finditer(recent_context_text):
            _add_candidate(m.group(1))
        for m in prefix_pat.finditer(recent_context_text):
            _add_candidate(m.group(1))

        named_artists = named_artists[:3]  # cap to avoid runaway queries

        # Resolve each detected artist to {id, name, genres}. The ID lets
        # us call /artists/{id}/top-tracks for verified picks (the only
        # way to disambiguate same-name artists like "Dave the UK rapper"
        # vs "Dave Matthews"). Genres are kept for the LLM's per-artist
        # context block.
        try:
            _client_for_lookup = get_spotify_client()
        except Exception:
            _client_for_lookup = None

        # name(case-preserved) -> {"id": str, "genres": [..]}.
        detected_artist_info: dict = {}
        for n in named_artists:
            key = n.lower()
            # Prefer the user's top-list data (we already have ID + genres).
            if key in artist_genre_map and (
                artist_genre_map[key] or key in artist_id_map
            ):
                detected_artist_info[n] = {
                    "id":     artist_id_map.get(key, ""),
                    "genres": (artist_genre_map.get(key) or [])[:5],
                }
                detected_artist_genres[n] = detected_artist_info[n]["genres"]
                continue
            # Fall back to a fresh Spotify artist-search lookup.
            if _client_for_lookup and _client_for_lookup._is_configured():
                try:
                    info = _client_for_lookup.find_artist(n)
                    if info and (info.get("id") or info.get("genres")):
                        detected_artist_info[n] = {
                            "id":     info.get("id") or "",
                            "genres": (info.get("genres") or [])[:5],
                        }
                        if info.get("genres"):
                            detected_artist_genres[n] = info["genres"][:5]
                except Exception:
                    pass

        # ── Verified artist top-tracks (highest-priority candidates) ────────
        # When we have an artist ID, fetch THAT EXACT artist's top tracks
        # and prepend them to the candidate pool. These bypass the
        # ambiguous track-search path entirely. Already-liked tracks are
        # filtered out so subsequent suggestions stay fresh.
        #
        # Engaged users (60+ likes) often have most of a named artist's
        # top 10 hearted already, leaving the priority bucket too thin
        # for the LLM to pick from. So when that happens, we supplement
        # by pulling top tracks from up to 4 of the named artist's
        # Spotify-canonical *related* artists. That keeps recommendations
        # in the same genre family without recycling already-liked songs.
        priority_candidates: list[dict] = []
        # Track which IDs we've already added so we don't double up across
        # the primary + related-artist supplements.
        priority_seen: set[str] = set()
        MIN_PRIORITY_FILL = 6  # supplement target: at least this many

        # Diagnostic logging — surfaces in your uvicorn terminal so we can
        # see which supplement paths fired and what each returned. Cheap to
        # leave on for now while we debug recommendation quality.
        print(f"[chat] named_artists={named_artists!r}")
        print(f"[chat] artist_id_map keys={list(artist_id_map.keys())}")
        print(f"[chat] detected_artist_info={ {k: {'id': v.get('id',''), 'genres': v.get('genres',[])} for k, v in detected_artist_info.items()} }")

        def _push_priority(track_dict: dict, seed_label: str):
            sid = track_dict.get("spotify_id") or ""
            if not sid or sid in priority_seen or sid in liked_ids:
                return
            priority_seen.add(sid)
            track_dict["mood"] = "from_artist"
            track_dict["seed_artist"] = seed_label
            priority_candidates.append(track_dict)

        if _client_for_lookup and _client_for_lookup._is_configured():
            for n, info in detected_artist_info.items():
                aid = (info.get("id") or "").strip()
                if not aid:
                    print(f"[chat] {n!r}: NO artist id, skipping verified bucket")
                    continue
                # 1) Primary: the named artist's own top tracks.
                try:
                    primary_tracks = _client_for_lookup.get_artist_top_tracks(aid)
                except Exception as exc:
                    print(f"[chat] {n!r} primary lookup failed: {exc}")
                    primary_tracks = []

                # If the user's top-artists ID gave us no tracks (sometimes
                # Spotify routes plays to a dormant same-name artist
                # profile), re-resolve the artist via name search and try
                # again. find_artist returns the most popular match,
                # which is what the user almost certainly meant.
                if not primary_tracks:
                    try:
                        info2 = _client_for_lookup.find_artist(n)
                    except Exception:
                        info2 = None
                    aid2 = (info2 or {}).get("id", "")
                    if aid2 and aid2 != aid:
                        print(f"[chat] {n!r} top-tracks empty for id={aid}; "
                              f"retrying with search-resolved id={aid2}")
                        try:
                            primary_tracks = _client_for_lookup.get_artist_top_tracks(aid2)
                        except Exception:
                            primary_tracks = []
                        # Also adopt the search-resolved ID + genres so
                        # the deep-catalog and genre-supplement paths
                        # below use the correct profile.
                        info["id"] = aid2
                        if info2 and info2.get("genres"):
                            info["genres"] = info2["genres"]
                            detected_artist_genres[n] = info2["genres"][:5]
                        aid = aid2

                pre_pri = len(priority_candidates)
                for t in primary_tracks:
                    _push_priority(t.as_dict(), seed_label=n)
                print(f"[chat] {n!r} primary: {len(primary_tracks)} fetched, {len(priority_candidates) - pre_pri} added (after liked-filter)")

                # 1b) Search-based fallback. Spotify's dev-mode policy
                # 403s /artists/{id}/top-tracks for new apps as of late
                # 2024, so when the primary path is empty we resort to
                # ``q=artist:"<name>"&type=track``. Spotify ranks by track
                # popularity, so the prominent artist with that name lands
                # at the top — we then post-filter by exact primary-artist
                # name match. Less precise than the artist endpoint, but
                # the only path that works for our app.
                if not primary_tracks:
                    try:
                        search_tracks = _client_for_lookup.search_artist_tracks_by_name(
                            name=n, limit=12, exclude_ids=liked_ids | priority_seen,
                        )
                    except Exception as exc:
                        print(f"[chat] {n!r} search fallback failed: {exc}")
                        search_tracks = []
                    pre_search = len(priority_candidates)
                    for t in search_tracks:
                        _push_priority(t.as_dict(), seed_label=n)
                    print(f"[chat] {n!r} search fallback: {len(search_tracks)} fetched, "
                          f"{len(priority_candidates) - pre_search} added")

                # 2) Deep-catalog supplement. Spotify's /related-artists
                # endpoint was deprecated for new apps in late 2024, so
                # we rely on two stable paths instead:
                #   (a) the named artist's albums → tracks (deep cuts,
                #       which is usually what a user wants when they say
                #       "more from X")
                #   (b) bare-genre Spotify track search using the
                #       artist's actual Spotify genre tags — this
                #       reliably surfaces same-genre prominent artists
                #       (Stormzy / Skepta / Headie One etc. for a
                #       "uk hip hop" seed).
                def _seed_count_for(name: str) -> int:
                    return sum(
                        1 for c in priority_candidates
                        if c.get("seed_artist", "").startswith(name)
                    )

                if _seed_count_for(n) < MIN_PRIORITY_FILL:
                    # 2a) Deep catalogue from the named artist's albums.
                    try:
                        deep_tracks = _client_for_lookup.get_artist_album_tracks(
                            aid, max_tracks=30,
                        )
                    except Exception as exc:
                        print(f"[chat] {n!r} deep-catalog failed: {exc}")
                        deep_tracks = []
                    pre_deep = _seed_count_for(n)
                    for t in deep_tracks:
                        if _seed_count_for(n) >= MIN_PRIORITY_FILL + 4:
                            break
                        _push_priority(t.as_dict(), seed_label=f"{n} (deep cut)")
                    print(f"[chat] {n!r} deep cuts: {len(deep_tracks)} fetched, {_seed_count_for(n) - pre_deep} added")

                if _seed_count_for(n) < MIN_PRIORITY_FILL:
                    # 2b) Genre-aligned tracks via bare-genre search.
                    artist_genres = info.get("genres") or []
                    if artist_genres:
                        genre_queries = [
                            f'genre:"{g}"'
                            for g in artist_genres[:3]
                            if g and g.strip()
                        ]
                        print(f"[chat] {n!r} genre supplement: queries={genre_queries}")
                        try:
                            genre_tracks = _client_for_lookup.paginated_search_tracks(
                                queries=genre_queries,
                                total=12,
                                exclude_ids=liked_ids | priority_seen,
                            )
                        except Exception as exc:
                            print(f"[chat] {n!r} genre supplement failed: {exc}")
                            genre_tracks = []
                        pre_gen = _seed_count_for(n)
                        for t in genre_tracks:
                            if _seed_count_for(n) >= MIN_PRIORITY_FILL + 6:
                                break
                            primary_artist = t.artists[0] if t.artists else ""
                            label = (
                                f"{n} (genre-aligned: {primary_artist})"
                                if primary_artist else
                                f"{n} (genre-aligned)"
                            )
                            _push_priority(t.as_dict(), seed_label=label)
                        print(f"[chat] {n!r} genre supplement: {len(genre_tracks)} fetched, {_seed_count_for(n) - pre_gen} added")
                    else:
                        print(f"[chat] {n!r} has no genres in info, skipping genre supplement")

                print(f"[chat] {n!r} TOTAL priority candidates: {_seed_count_for(n)}")

        # Pre-pend artist-verified picks to the pool so the LLM sees them
        # first when grouping by mood. We dedupe and seed the IDs into
        # ``seen_ids`` later so the mood-loop won't re-collect them.
        if priority_candidates:
            spotify_candidates.extend(priority_candidates)

        client = get_spotify_client()
        if client._is_configured():
            # Order moods so the user's dominant taste comes first — that
            # subset is naturally over-represented in the pool, which
            # matches how the user actually listens, but the other three
            # moods are still present so follow-ups like "give me sadder
            # picks" don't fall off a cliff.
            ordered_moods = ["energetic", "calm", "happy", "sad"]
            if profile.has_signal():
                dom = (profile.dominant_emotion() or "").lower()
                if dom in ordered_moods:
                    ordered_moods.remove(dom)
                    ordered_moods.insert(0, dom)

            # Seed the dedupe set with priority-artist IDs so the mood
            # search loop doesn't re-collect them (the LLM should see
            # them once, in the priority bucket, not duplicated under a
            # mood bucket).
            seen_ids: set[str] = {
                (c.get("spotify_id") or "")
                for c in priority_candidates
                if c.get("spotify_id")
            }
            per_mood_target = 6  # ~24 total candidates across 4 moods

            # Helper: dedupe a list while preserving first-seen order.
            def _dedup(seq):
                out, seen = [], set()
                for v in seq:
                    if v and v not in seen:
                        seen.add(v)
                        out.append(v)
                return out

            for mood in ordered_moods:
                mq = mood_to_query(discrete_emotion=mood)

                # Build a richer query pool per mood. Order matters:
                # detected-artist clauses go first (highest priority for
                # the user's latest message), then user's top artists,
                # then artist-derived genres, then plain mood keywords as
                # a safety net so the rail always fills.
                queries: list[str] = []

                # 1) Anchor on artists the user just named (and the
                #    genres of those artists, so non-top picks stay in
                #    the right family).
                if named_artists and mq.queries:
                    for i, artist in enumerate(named_artists):
                        mood_word = mq.queries[i % len(mq.queries)]
                        queries.append(f'artist:"{artist}" {mood_word}')
                        for j, g in enumerate(detected_artist_genres.get(artist, [])[:3]):
                            mw = mq.queries[(i + j + 1) % len(mq.queries)]
                            queries.append(f'genre:"{g}" {mw}')

                # 2) User's top artists.
                if seed_artists and mq.queries:
                    for i, artist in enumerate(seed_artists):
                        mood_word = mq.queries[i % len(mq.queries)]
                        queries.append(f'artist:"{artist}" {mood_word}')

                # 3) User's top genres (across top artists).
                if seed_genres and mq.queries:
                    for i, genre in enumerate(seed_genres):
                        mood_word = mq.queries[(i + 1) % len(mq.queries)]
                        queries.append(f'genre:"{genre}" {mood_word}')

                # 4) Plain mood pool — guaranteed fallback.
                queries.extend(mq.queries)
                queries = _dedup(queries)

                try:
                    tracks = client.paginated_search_tracks(
                        queries=queries,
                        total=per_mood_target,
                        exclude_ids=liked_ids | seen_ids,
                    )
                    for t in tracks:
                        if not t.spotify_id or t.spotify_id in seen_ids:
                            continue
                        seen_ids.add(t.spotify_id)
                        d = t.as_dict()
                        d["mood"] = mood  # tag so the LLM can filter
                        spotify_candidates.append(d)
                except Exception:
                    # One mood's search failing shouldn't take the rest down.
                    continue
    except Exception as exc:
        # Spotify down / unconfigured / network blip — chat still answers
        # from the indexed dataset, just without the live recommendation
        # pool. Log so we can see the rate, but never fail the chat call.
        print("⚠️ Spotify candidate fetch failed for /chat:", exc)
        spotify_candidates = []

    # Bolt the artists the user named (with their fetched genre tags)
    # into the spotify_context so the prompt's "per-artist genre" block
    # surfaces them — even if the artist isn't in the user's top list.
    enriched_spotify_context = dict(req.spotify_context or {}) if req.spotify_context else {}
    if detected_artist_genres:
        existing_ag = list(enriched_spotify_context.get("artist_genres") or [])
        existing_names = {(a.get("name") or "").lower() for a in existing_ag if isinstance(a, dict)}
        for name, genres in detected_artist_genres.items():
            if name.lower() not in existing_names:
                existing_ag.append({"name": name, "genres": genres})
                existing_names.add(name.lower())
        enriched_spotify_context["artist_genres"] = existing_ag
        enriched_spotify_context["named_in_message"] = list(detected_artist_genres.keys())

    system = build_system_prompt(
        profile=profile,
        index=index,
        last_analyzed_track=req.last_track_meta,
        spotify_context=enriched_spotify_context or req.spotify_context,
        spotify_candidates=spotify_candidates,
    )

    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "system": system,
        "messages": req.messages,
    }

    # No key configured at all → return the canned safe fallback so the
    # public demo doesn't crash for visitors without API access.
    if not api_key:
        return {"response": _DEMO_FALLBACK_RESPONSE.strip()}

    # Key present → call Anthropic and surface any failure to the frontend
    # so the user sees the real reason (invalid key, expired credits, model
    # not available) instead of a misleading "fallback" message.
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=20,
        )
    except Exception as exc:
        print("⚠️ Network error calling Anthropic:", exc)
        raise HTTPException(status_code=502, detail=f"Could not reach Anthropic: {exc}")

    if response.status_code == 200:
        data = response.json()
        text = "".join(
            b.get("text", "")
            for b in data.get("content", [])
            if b.get("type") == "text"
        )

        # Pull out the Spotify track IDs the LLM linked to in its response,
        # in order, so the frontend can render them as visual cards (album
        # art + title + artist + Open-in-Spotify link) instead of leaving
        # the user to parse markdown URLs. The LLM is instructed to only
        # link to URLs from the candidate pool, so each ID we find here
        # should match one of ``spotify_candidates``.
        import re as _re
        cited_ids: list[str] = []
        seen: set[str] = set()
        for match in _re.finditer(
            r"open\.spotify\.com/track/([A-Za-z0-9]+)",
            text,
        ):
            sid = match.group(1)
            if sid and sid not in seen:
                seen.add(sid)
                cited_ids.append(sid)

        cand_by_id = {c.get("spotify_id"): c for c in (spotify_candidates or [])}
        tracks_out = [cand_by_id[sid] for sid in cited_ids if sid in cand_by_id]

        return {"response": text, "tracks": tracks_out}

    # Non-200: log and bubble up. Anthropic returns JSON like
    # {"error": {"type": "...", "message": "..."}}. We surface message + type
    # so the frontend's existing "credit"-substring check still works for
    # exhausted-balance cases.
    body = response.text
    print(f"⚠️ Anthropic returned {response.status_code}: {body}")
    try:
        err = response.json().get("error", {})
        detail = err.get("message") or body
    except Exception:
        detail = body
    raise HTTPException(status_code=response.status_code, detail=detail)


_DEMO_FALLBACK_RESPONSE = """
Auralis Insight:

The chat is running in demo-safe fallback mode because no Anthropic API key
is configured. Set ANTHROPIC_API_KEY in your backend environment to enable
fully grounded chat responses.

At this stage, recommendations are driven by:
- Acoustic similarity (cosine similarity)
- Emotion alignment (calm, energetic, happy, sad)
"""


@app.post("/spotify/search-by-mood")
def spotify_search_by_mood(req: SpotifyMoodRequest):
    """Surface real Spotify tracks for a (valence, arousal) mood point.

    The frontend passes the same coordinates the circumplex UI uses, so the
    same tap that shows the user where they are emotionally also fills the
    recommendation rail with live Spotify results.

    Two improvements over a naive single search:

    1. **Pagination** — Spotify's Client-Credentials search endpoint started
       returning 400 "Invalid limit" for limit > 10 in early 2026. We now
       fan out across multiple offsets so playlists can comfortably reach
       30 tracks without ever asking Spotify for more than 10 at a time.
    2. **Taste blending** — when the frontend forwards the user's top
       Spotify artists (and/or genres) as ``seed_artists``, those names are
       woven into the Search query pool so the playlist reflects the
       listener's actual taste rather than only generic mood keywords.
    """

    mq = mood_to_query(
        valence=req.valence,
        arousal=req.arousal,
        discrete_emotion=req.discrete_emotion,
    )

    # Build the query pool the paginated search will rotate through.
    # Order: explicit override (highest priority) → artist-blended queries →
    # plain mood keywords (always included as a safety net so the playlist
    # still fills up if every artist clause comes back empty).
    queries: list[str] = []

    if req.query_override and req.query_override.strip():
        queries.append(req.query_override.strip())

    mood_keywords = list(mq.queries)

    # Take up to 5 seed artists — past that the Search query string gets
    # long enough to risk Spotify truncating it. Sample one mood phrase per
    # artist so different fan-out calls explore different mood facets of
    # the same artist.
    seed_artists = [a.strip() for a in (req.seed_artists or []) if isinstance(a, str) and a.strip()]
    seed_artists = seed_artists[:5]
    if seed_artists and mood_keywords:
        for artist in seed_artists:
            # `artist:"X"` is a Spotify Search field filter. Pairing it with a
            # mood keyword surfaces tracks that match BOTH the artist's catalogue
            # AND the requested mood, which is exactly the blend the user asked
            # for ("analyze both my Spotify and Auralis profile").
            mood_word = mood_keywords[len(queries) % len(mood_keywords)]
            queries.append(f'artist:"{artist}" {mood_word}')

    # Genre seeds (optional). One blended query per genre keeps the request
    # space small but still benefits from Spotify's genre tagging.
    seed_genres = [g.strip() for g in (req.seed_genres or []) if isinstance(g, str) and g.strip()]
    seed_genres = seed_genres[:3]
    if seed_genres and mood_keywords:
        for genre in seed_genres:
            mood_word = mood_keywords[(len(queries) + 1) % len(mood_keywords)]
            queries.append(f'genre:"{genre}" {mood_word}')

    # Always append the plain mood pool last — guarantees we still return
    # results when every artist/genre clause comes back empty.
    queries.extend(mood_keywords)

    # Fold the user's previously-hearted Spotify tracks into ``exclude_ids``
    # so the playlist surfaces new material instead of resurfacing songs
    # the user already knows. Combine with any explicit excludes the
    # frontend supplies (e.g. tracks already pushed in this session).
    auralis_liked: set[str] = set()
    try:
        for entry in get_profile().interaction_log:
            track = entry.get("track", "")
            if track.startswith("spotify:track:") and entry.get("feedback") == "like":
                sid = track.split(":", 2)[-1]
                if sid:
                    auralis_liked.add(sid)
    except Exception:
        # Profile read failure shouldn't block playlist generation.
        auralis_liked = set()

    explicit_excludes = {
        e for e in (req.exclude_ids or [])
        if isinstance(e, str) and e.strip()
    }
    excluded = auralis_liked | explicit_excludes

    try:
        client = get_spotify_client()
        tracks = client.paginated_search_tracks(
            queries=queries,
            total=req.limit,
            exclude_ids=excluded,
            shuffle_seed=req.shuffle_seed,
        )
    except SpotifyError as exc:
        # 503 lets the frontend fall back to local recommendations cleanly.
        raise HTTPException(status_code=503, detail=str(exc))

    blended = bool(seed_artists or seed_genres)
    rationale = mq.rationale
    if blended:
        seed_label_parts = []
        if seed_artists:
            seed_label_parts.append(f"top artists ({', '.join(seed_artists[:3])}{'...' if len(seed_artists) > 3 else ''})")
        if seed_genres:
            seed_label_parts.append(f"genres ({', '.join(seed_genres)})")
        rationale = (
            f"{mq.rationale} Blended with your Spotify "
            f"{' and '.join(seed_label_parts)}."
        )
    if auralis_liked:
        rationale = (
            f"{rationale} Filtering out {len(auralis_liked)} track"
            f"{'s' if len(auralis_liked) != 1 else ''} you've already liked."
        )

    return {
        "query":      queries[0] if queries else "",
        "queries":    queries,
        "blended":    blended,
        "quadrant":   mq.quadrant,
        "rationale":  rationale,
        "tracks":     [t.as_dict() for t in tracks],
    }


# ── Synthetic emotion scores for Spotify-track feedback ──────────────────────
# We don't have audio features (and therefore no MFCC vector or measured
# emotion classifier output) for tracks pulled in via Spotify Search. To
# still let the user's likes nudge their Auralis profile, we map the mood
# they were browsing onto a small soft-distribution over the four emotion
# labels. The weights aren't a clinical model — they're a calibrated
# fallback so a Spotify "like in the energetic rail" pulls the user's
# affinity centroid toward energetic without zeroing the other axes.
_MOOD_TO_SOFT_SCORES = {
    "energetic": {"energetic": 0.70, "happy": 0.30, "calm": 0.10, "sad": 0.10},
    "happy":     {"happy": 0.70, "energetic": 0.40, "calm": 0.20, "sad": 0.05},
    "calm":      {"calm": 0.70, "happy": 0.30, "sad": 0.20, "energetic": 0.10},
    "sad":       {"sad": 0.70, "calm": 0.40, "happy": 0.10, "energetic": 0.05},
}


@app.post("/spotify/feedback")
def spotify_feedback(req: SpotifyFeedbackRequest):
    """Record a like/dislike on a Spotify track in the Auralis profile.

    This is the bridge between the Spotify cards on the For You / Playlist
    rails and the user's local taste profile. Spotify's `PUT /me/tracks`
    write API is currently blocked for our Dev-Mode app, so the heart
    button on a Spotify card calls this endpoint instead — the like still
    counts toward the user's Auralis-side preferences and shows up in the
    Profile page's Interaction History.
    """

    emotion = (req.discrete_emotion or "calm").strip().lower()
    if emotion not in _MOOD_TO_SOFT_SCORES:
        emotion = "calm"

    profile = get_profile()
    profile = record_spotify_feedback(
        profile=profile,
        spotify_id=req.spotify_id,
        track_name=req.track_name,
        artists=list(req.artists or []),
        label=req.label,
        emotion_scores=_MOOD_TO_SOFT_SCORES[emotion],
        emotion_label=emotion,
        profile_save_path=PROFILE_PATH,
    )
    return feedback_summary(profile)


@app.get("/spotify/health")
def spotify_health():
    """Quick liveness check the frontend can poll before showing Spotify UI."""
    client = get_spotify_client()
    if not client._is_configured():
        return {"status": "unconfigured"}
    try:
        client._get_token()
        return {"status": "ok"}
    except SpotifyError as exc:
        return {"status": "error", "detail": str(exc)}