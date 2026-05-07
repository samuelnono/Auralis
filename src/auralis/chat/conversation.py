"""
Auralis - Conversation Manager
--------------------------------
Builds a grounded system prompt from the user's live profile and
song index so the LLM always reasons from real acoustic data,
never from hallucinated music knowledge.
"""

from __future__ import annotations

import json
from typing import List, Dict, Optional

from src.auralis.preference.profile import UserProfile


def build_system_prompt(
    profile: UserProfile,
    index: Optional[List[Dict]] = None,
    last_analyzed_track: Optional[Dict] = None,
    spotify_context: Optional[Dict] = None,
    spotify_candidates: Optional[List[Dict]] = None,
) -> str:
    """
    Build a fully grounded system prompt for the Auralis chat assistant.

    Injects:
      - User's emotion affinity profile (from local ratings)
      - Dominant taste and rating history summary
      - Top 8 indexed songs (name + emotion) for recommendation context
      - Last analyzed track metadata (if available)
      - The user's Spotify listening highlights (if connected) so the model
        can reason from real listening history, not just locally-rated tracks
      - A small pool of fresh Spotify-search candidates (track name, artists,
        Spotify URL) the LLM can recommend with playable links instead of
        falling back to the dataset. Pulled per-message in the backend so the
        recommendation pool reflects the user's *current* mood, not a stale
        cache.
    """

    # ── Profile summary ──────────────────────────────────────────────────────
    if profile.has_signal():
        affinity = profile.emotion_affinity
        dominant = profile.dominant_emotion()
        affinity_str = ", ".join(
            f"{k}: {v:.2f}" for k, v in sorted(affinity.items(), key=lambda x: -x[1])
        )
        profile_block = (
            f"The user has rated {profile.total_likes} track(s) as liked "
            f"and {profile.total_dislikes} as disliked.\n"
            f"Their dominant emotion preference is: {dominant}.\n"
            f"Emotion affinity scores (0-1): {affinity_str}."
        )
    else:
        profile_block = "The user has not rated any tracks yet. No preference profile exists."

    # ── Index summary (top 8 tracks by name + emotion + mood) ────────────────
    if index:
        track_lines = []
        for row in index[:8]:
            from pathlib import Path
            name = Path(row.get("path", "unknown")).stem
            emotion = row.get("predicted_emotion") or row.get("emotion", "unknown")
            calm      = float(row.get("calm", 0)      or 0)
            energetic = float(row.get("energetic", 0) or 0)
            happy     = float(row.get("happy", 0)     or 0)
            sad       = float(row.get("sad", 0)       or 0)

            # Optional continuous-mood coordinates (populated by v1.1+ indexer).
            mood_suffix = ""
            try:
                v = float(row["valence"])
                a = float(row["arousal"])
                q = row.get("quadrant", "")
                n = row.get("nuanced_tag", "")
                mood_suffix = (
                    f", mood=({v:+.2f}, {a:+.2f}) [{q or 'unknown'}"
                    + (f" / {n}" if n else "")
                    + "]"
                )
            except (KeyError, TypeError, ValueError):
                pass

            track_lines.append(
                f"  - {name}: dominant={emotion}, "
                f"calm={calm:.2f}, energetic={energetic:.2f}, "
                f"happy={happy:.2f}, sad={sad:.2f}{mood_suffix}"
            )
        index_block = "Available indexed tracks:\n" + "\n".join(track_lines)
    else:
        index_block = "No indexed tracks are available yet."

    # ── Last analyzed track ──────────────────────────────────────────────────
    if last_analyzed_track:
        from pathlib import Path
        name = Path(last_analyzed_track.get("path", "unknown")).stem
        emotion = last_analyzed_track.get("predicted_emotion") or last_analyzed_track.get("emotion", "unknown")
        tempo   = last_analyzed_track.get("tempo", "N/A")
        rms     = last_analyzed_track.get("rms_mean", "N/A")
        centroid = last_analyzed_track.get("spectral_centroid_mean", "N/A")
        duration = last_analyzed_track.get("duration_sec", "N/A")

        mood = last_analyzed_track.get("mood") or {}
        if mood:
            mood_line = (
                f"  Dimensional mood: valence={mood.get('valence', 0):+.2f}, "
                f"arousal={mood.get('arousal', 0):+.2f}, "
                f"quadrant={mood.get('quadrant', 'unknown')}, "
                f"nuanced={mood.get('nuanced_tag', 'unknown')}"
            )
        else:
            mood_line = "  Dimensional mood: not available"

        track_block = (
            f"The user most recently analyzed a track called '{name}'.\n"
            f"  Predicted emotion (discrete): {emotion}\n"
            f"  Tempo: {round(float(tempo), 1) if tempo != 'N/A' else 'N/A'} BPM\n"
            f"  RMS energy: {round(float(rms), 4) if rms != 'N/A' else 'N/A'}\n"
            f"  Spectral centroid: {round(float(centroid), 1) if centroid != 'N/A' else 'N/A'} Hz\n"
            f"  Duration: {round(float(duration), 1) if duration != 'N/A' else 'N/A'} sec\n"
            f"{mood_line}"
        )
    else:
        track_block = "No track has been analyzed in this session yet."

    # ── Spotify listening highlights (only if the user is connected) ─────────
    if spotify_context:
        top_artists   = spotify_context.get("top_artists") or []
        top_tracks    = spotify_context.get("top_tracks") or []
        top_genres    = spotify_context.get("top_genres") or []
        artist_genres = spotify_context.get("artist_genres") or []
        display       = spotify_context.get("display_name") or "the user"

        artist_line = (
            ", ".join(a for a in top_artists[:8] if a)
            if top_artists else "(no top artists yet)"
        )
        genre_line = (
            ", ".join(g for g in top_genres[:6] if g)
            if top_genres else "(no genre tags available)"
        )
        # Per-artist genre map so the LLM can reason about *which* genre
        # family an artist belongs to when the user asks for "X-style".
        # Keeps the model from picking country tracks for a UK rap seed.
        artist_genre_lines = []
        for ag in artist_genres[:8]:
            name = (ag.get("name") or "").strip()
            genres = ", ".join(g for g in (ag.get("genres") or []) if g)
            if name:
                artist_genre_lines.append(
                    f"  - {name}: {genres or 'no Spotify genre tags'}"
                )
        artist_genre_block = (
            "\n".join(artist_genre_lines)
            if artist_genre_lines else "  (no per-artist genre tags available)"
        )

        track_lines = []
        for t in top_tracks[:10]:
            name    = (t.get("name") or "").strip()
            artists = ", ".join(a for a in (t.get("artists") or []) if a)
            if name:
                track_lines.append(f"  - {name}" + (f" — {artists}" if artists else ""))

        track_block_str = "\n".join(track_lines) if track_lines else "  (no top tracks yet)"
        named_in_message = spotify_context.get("named_in_message") or []
        named_block = (
            f"\nArtists the user explicitly referenced in this message: "
            f"{', '.join(named_in_message)}.\n"
            f"⇒ Pick ONLY tracks whose primary artist sits in the same genre "
            f"family as the named artists (per the per-artist genre tags above). "
            f"Reject any candidate whose genre family is unrelated, even if its "
            f"name vibes with the mood keyword. If the pool doesn't have a "
            f"perfect-genre match, pick the closest neighbor (e.g. for hip-hop "
            f"seeds, R&B / soul / alt-rap is fair; country, ambient, easy "
            f"listening are not)."
        ) if named_in_message else ""

        # Note: we deliberately do NOT include the user's "Top tracks"
        # list anymore. Including it kept tempting the LLM to recycle
        # those titles as recommendations even when the rules said
        # otherwise — top tracks are by definition things the user
        # already listens to, so picking them is lazy and contradicts
        # the "always offer something new" rule. Top *artists* (with
        # genre tags) carry the same taste signal without the temptation.
        spotify_block = (
            f"{display} is signed into Spotify, so you have access to their real listening history.\n"
            f"Top artists (medium-term, ~6 months): {artist_line}\n"
            f"Top genres across those artists: {genre_line}\n"
            f"Per-artist genre tags (use this when the user asks for 'X-style' so "
            f"the picks stay in the right musical neighborhood):\n{artist_genre_block}"
            f"{named_block}\n\n"
            f"When the user asks about their taste, music identity, or what to listen to, "
            f"reason from BOTH their locally-rated emotion profile AND this Spotify history. "
            f"They are complementary — the local profile captures emotional affinity, the "
            f"top-artists list captures actual play frequency, and the genre tags constrain the "
            f"musical universe the recommendations should live in."
        )
    else:
        spotify_block = (
            "The user is not connected to Spotify. Reason about taste only from the "
            "local rating profile and analyzed tracks above."
        )

    # ── Spotify recommendation pool (fresh per message) ─────────────────────
    # Candidates are pulled per-message with mood tags so the LLM can
    # answer mood-specific follow-ups ("give me sadder picks") without
    # falling back to the indexed dataset. We group by mood here so the
    # model sees the structure clearly.
    if spotify_candidates:
        by_mood: Dict[str, List[Dict]] = {}
        for c in spotify_candidates:
            mood = (c.get("mood") or "uncategorised").lower()
            by_mood.setdefault(mood, []).append(c)

        # Render in a mood-prioritised order. The "from_artist" bucket
        # always leads — those are tracks fetched directly from a named
        # artist's Spotify catalogue (verified, no ambiguity) and should
        # be the LLM's first-choice picks when the user mentioned an
        # artist by name. The four mood buckets follow.
        canonical = ["from_artist", "energetic", "calm", "happy", "sad", "uncategorised"]
        moods_seen = list(by_mood.keys())
        # Preserve insertion order from the backend (which puts the user's
        # dominant taste first), but ensure we cover everything we have.
        ordered_moods = list(dict.fromkeys(["from_artist"] + moods_seen + canonical))
        ordered_moods = [m for m in ordered_moods if m in by_mood]

        sections: List[str] = []
        for mood in ordered_moods:
            cand_lines = []
            for c in by_mood[mood]:
                name = (c.get("name") or "").strip()
                if not name:
                    continue
                artists = ", ".join(a for a in (c.get("artists") or []) if a)
                url = c.get("external_url") or ""
                seed_artist = (c.get("seed_artist") or "").strip()
                line = f"  - {name}" + (f" — {artists}" if artists else "")
                if seed_artist:
                    line += f" [from {seed_artist}'s catalogue]"
                if url:
                    line += f" ({url})"
                cand_lines.append(line)
            if cand_lines:
                heading = (
                    "Verified picks from named artists' catalogues"
                    if mood == "from_artist" else mood.title()
                )
                sections.append(f"### {heading}\n" + "\n".join(cand_lines))

        candidates_block = (
            "Live Spotify-search candidates, blended with the user's top "
            "artists and grouped by mood. Already-hearted tracks are filtered "
            "out so suggestions stay fresh. Pick from whichever mood subset "
            "best matches the user's request — you have plenty of options "
            "across all four emotion buckets:\n\n"
            + "\n\n".join(sections)
        ) if sections else "No fresh Spotify candidates were available for this turn."
    else:
        candidates_block = (
            "No Spotify recommendation pool available for this turn — fall back to "
            "the indexed song collection below when recommending, and tell the user "
            "Spotify candidates couldn't be fetched."
        )

    # ── Full system prompt ───────────────────────────────────────────────────
    return f"""You are Auralis, an intelligent music assistant that combines acoustic signal processing with emotional awareness to help users discover and understand music.

You are grounded in real data — all your responses about tracks, emotions, and recommendations must reference the acoustic features, profile data, or Spotify candidates provided below. Do not invent music knowledge or pretend to know songs you haven't been given data about.

## User Preference Profile (from local ratings)
{profile_block}

## Spotify Listening History
{spotify_block}

## Spotify Recommendation Pool (this turn)
{candidates_block}

## Song Index (research dataset, fallback only)
{index_block}

## Recently Analyzed Track
{track_block}

## Your Capabilities
1. **Recommend tracks** — draw from the Spotify recommendation pool above when available. Only fall back to the indexed dataset when no Spotify pool exists, and say so.
2. **Explain recommendations** briefly, referencing the user's profile or acoustic features.
3. **Answer questions** about any analyzed track's emotional and acoustic properties.
4. **Discuss music taste** using the emotion affinity profile and Spotify history as context.

## Emotional Representation
Auralis uses two mood layers: **discrete** (calm / energetic / happy / sad) and **dimensional** (valence × arousal on [-1, 1], with quadrant + nuanced sub-tag). Reference whichever best answers the question.

## Output Format Rules — IMPORTANT
The frontend renders any Spotify URL you link to as a visual card with album art, track name, and artist. So:
- Do NOT repeat the track name and artist in prose after the link — the card already shows them.
- Each pick = one markdown link `[Track — Artist](url)` followed by AT MOST one short clause (≤ 12 words) explaining why it fits. Example: `[Marvellous — Dave](url) — matches your energetic + reflective lean.`
- Default to **3 picks**. Only return more if the user explicitly asks for more.
- Keep the framing sentence to ONE line. No headers, no emojis, no "Let's find your next track" preamble. Open the response with the picks immediately.
- After the picks, optionally one short follow-up question (≤ 10 words). Nothing more.
- Total response budget: ≤ 100 words.

## Recommendation Behaviour
- **"X coded", "X-style", "X vibes", "X-esque", "like X" all mean: recommend tracks in the same musical neighborhood as artist X.** Never read these phrases as a track title. They are *always* stylistic requests.
- **Verified-artist bucket comes first.** When you see a "Verified picks from named artists' catalogues" section above, those tracks were pulled directly from the named artist's Spotify catalogue (using their canonical artist ID, not a fuzzy text search). They are the highest-confidence picks — open with them whenever the user named an artist.
- **Pick ONLY from the Spotify Recommendation Pool above.** The pool's tracks are the only ones whose Spotify URLs you actually know. Recommending anything else means inventing a URL — strictly forbidden. If a track from your training data feels relevant but isn't in the pool, do NOT cite it.
- **The user's "Top tracks" list is FYI only — never recommend from it.** Those are tracks they already listen to. The pool was deliberately constructed to exclude already-liked Spotify tracks from their Auralis profile, so anything you cite from outside the pool is almost certainly something they've already heard.
- The pool is grouped by mood (Energetic / Calm / Happy / Sad) plus the verified-artist bucket. When the user asks for a particular feeling, pick from that mood's bucket first.
- **Genre alignment is non-negotiable when the user names an artist.** Look up that artist in the per-artist genre tags above and only pick pool candidates from the same genre family. A UK-rap seed should never produce a country pick. If the pool has nothing in that genre family for the requested mood, pick from a related family in the same broader space (e.g. for hip-hop seeds, R&B / soul / alt-rap is fair game; country, easy listening, ambient instrumentals are not).
- If the pool genuinely has nothing in the requested genre family even loosely, say so honestly in one sentence ("nothing in your hip-hop pool this turn — try refreshing") and offer the closest pool match anyway. **Do not recommend tracks not in the pool.**
- **Artist diversity:** every 3-pick set must feature **at least 3 distinct primary artists** unless the user explicitly asks for "more from one artist". When picking from the verified-artist bucket, mixing in 1-2 cross-artist neighbors from related genre buckets is encouraged.
- **Prefer tracks with vocals/lyrics by default.** Skip candidates whose name signals instrumental ("Easy Listening Piano", "Bossa Nova with Lead Guitar", "Soothing Strings", "Lofi Study Mix") unless the user explicitly asks for instrumentals.
- Use the indexed dataset only when the user explicitly asks about acoustic features of an indexed track, or when no Spotify pool is available at all.

## Other Rules
- For indexed-dataset answers, ground explanations in acoustic data: "based on its spectral centroid of X Hz" not "this song sounds bright".
- Never fabricate feature values, emotion scores, mood coordinates, or Spotify URLs. Only link to URLs from the pool above.
"""


def format_history_for_api(history: List[Dict]) -> List[Dict]:
    """
    Convert Auralis chat history format to Anthropic API messages format.
    history items: {"role": "user"|"assistant", "content": str}
    """
    return [{"role": item["role"], "content": item["content"]} for item in history]
