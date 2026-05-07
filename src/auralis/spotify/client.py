"""Thin Spotify Web API wrapper used for app-level Search requests.

Why this looks the way it does
------------------------------
Spotify's Feb 2026 Developer Mode tightening means we cannot batch-fetch audio
features any more, and the curated ``/recommendations`` seed endpoint is gone.
What we *can* still do, with a plain Client Credentials token, is hit
``GET /v1/search`` for tracks and surface their preview URLs and Spotify links.

This module wraps that single use-case. It caches the access token in process
memory until it nears expiry, retries once on 401, and degrades gracefully when
Spotify is unreachable so the rest of the backend keeps responding.

Configuration is purely environment-driven so secrets stay out of the repo.
The Client ID is treated as public; the Client Secret must come from a Fly.io
secret in production and from a local .env file during development.
"""

from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import requests


SPOTIFY_TOKEN_URL  = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

# Spotify's Search docs advertise limit up to 50, but as of early 2026 we
# started seeing intermittent 400 "Invalid limit" responses for any request
# with limit > 10 against Client-Credentials tokens. Until that's clarified,
# the safest cap is 10 per call. Callers that want more results should use
# `paginated_search_tracks()` which fans out across multiple offsets and
# dedupes the combined output.
DEFAULT_LIMIT     = 10
MAX_LIMIT         = 10
MAX_TOTAL_RESULTS = 100  # ceiling for paginated_search_tracks combined output


# ── Stock-music / instrumental filtering ─────────────────────────────────────
# Spotify search keyword-matches mood phrases like "uplifting pop" or "chill
# acoustic", which means tracks from royalty-free / production-music labels
# leak into the candidate pool every time. The user's recommendation surface
# is for *songs they would actually listen to*, not background music for a
# YouTube video, so we filter those out at the source.
#
# This denylist matches the ARTIST name (case-insensitive substring). It is
# intentionally narrow — extending it doesn't risk losing real-music artists
# because each entry is a known stock-music brand or naming convention.
_STOCK_ARTIST_DENY = {
    "ashamaluevmusic",
    "bensound",
    "audio network",
    "epidemic sound",
    "lofi girl",
    "lofi study",
    "background music",
    "stock music",
    "royalty free",
    "no copyright",
    "ncm music",
    "ncs",
    "study music",
    "relaxing music",
    "meditation music",
    "spa music",
    "yoga music",
    "sleep music",
    "white noise",
    "easy listening",
    "smooth jazz orchestra",
    "instrumental music",
    "happy music",
    "chill music",
    "calm music",
    "music for studying",
    "music for sleep",
    "music for meditation",
}

# Track-name substrings that flag the track as production music or
# instrumental filler. Same case-insensitive substring rule.
_STOCK_TITLE_DENY = (
    "background music",
    "stock music",
    "royalty free",
    "no copyright",
    "music for studying",
    "music for sleep",
    "music for meditation",
    "study music",
    "relaxing music",
    "meditation music",
    "spa music",
    "yoga music",
    "sleep music",
    "lofi study",
    "white noise",
    "instrumental version",
    "karaoke version",
)


def is_stock_or_instrumental(track: "SpotifyTrack") -> bool:
    """Best-effort detection of royalty-free / instrumental filler.

    Used by the candidate-pool fetch in /chat and /spotify/search-by-mood
    to keep stock-music tracks out of recommendations. The user can still
    explicitly ask for instrumentals — that path bypasses this filter
    upstream by skipping the call.
    """
    name_lc = (track.name or "").lower()
    for needle in _STOCK_TITLE_DENY:
        if needle in name_lc:
            return True
    for artist in (track.artists or []):
        a_lc = (artist or "").lower()
        if not a_lc:
            continue
        # Substring-match against the curated denylist.
        for needle in _STOCK_ARTIST_DENY:
            if needle in a_lc:
                return True
        # Heuristic: artists whose name ends with bare "Music" /
        # "Studios" / "Productions" are overwhelmingly stock-music
        # accounts (real artists who happen to use these words almost
        # always have something else attached). Tightened with
        # word-boundary checks to avoid false positives like "MUSIC SOULchild".
        for suffix in (" music", " studios", " productions", " sound", " fx"):
            if a_lc.endswith(suffix):
                return True
    return False

# Refresh the token a minute before the real expiry so a slow request can't
# straddle the boundary. Spotify tokens are typically valid for one hour.
TOKEN_REFRESH_SAFETY_SEC = 60


class SpotifyError(RuntimeError):
    """Anything that prevented us from returning real Spotify data."""


@dataclass
class SpotifyTrack:
    """A normalised track payload that the rest of the app can rely on.

    We keep the schema small so the frontend has stable field names regardless
    of how Spotify reshapes its raw response in the future.
    """

    spotify_id: str
    name: str
    artists: list[str]
    album: str
    album_art_url: Optional[str]
    preview_url: Optional[str]
    external_url: str
    duration_ms: int
    explicit: bool

    def as_dict(self) -> dict:
        return {
            "spotify_id":    self.spotify_id,
            "name":          self.name,
            "artists":       self.artists,
            "album":         self.album,
            "album_art_url": self.album_art_url,
            "preview_url":   self.preview_url,
            "external_url":  self.external_url,
            "duration_ms":   self.duration_ms,
            "explicit":      self.explicit,
        }


@dataclass
class _CachedToken:
    access_token: str
    expires_at: float


@dataclass
class SpotifyClient:
    """Singleton-style client. One per process is enough for app-level reads."""

    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    request_timeout_sec: float = 8.0
    _token: Optional[_CachedToken] = field(default=None, init=False, repr=False)

    # Shared across all SpotifyClient instances — caches the artist→genre
    # lookups the chat backend triggers when the user says "X-style". Marked
    # ClassVar so @dataclass treats it as a class attribute, not a field
    # (which would otherwise reject the mutable default).
    _artist_genre_cache: ClassVar[dict] = {}

    def __post_init__(self) -> None:
        # Allow zero-arg construction in tests by reading env at call time too,
        # but resolving here gives us an early failure if both are missing.
        self.client_id     = self.client_id     or os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = self.client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")

    # -- Auth --------------------------------------------------------------

    def _is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _fetch_app_token(self) -> _CachedToken:
        """Run the Client Credentials flow once and return a cached token."""

        if not self._is_configured():
            raise SpotifyError(
                "Spotify credentials are not set. "
                "Provide SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET."
            )

        # Spotify wants HTTP Basic with the colon-joined ID:SECRET, base64'd.
        creds = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        auth_header = base64.b64encode(creds).decode("ascii")

        try:
            response = requests.post(
                SPOTIFY_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type":  "application/x-www-form-urlencoded",
                },
                data={"grant_type": "client_credentials"},
                timeout=self.request_timeout_sec,
            )
        except requests.RequestException as exc:
            raise SpotifyError(f"Could not reach Spotify token endpoint: {exc}") from exc

        if response.status_code != 200:
            raise SpotifyError(
                f"Spotify token request failed ({response.status_code}): {response.text}"
            )

        body = response.json()
        access_token = body.get("access_token")
        expires_in   = int(body.get("expires_in", 3600))
        if not access_token:
            raise SpotifyError("Spotify token response did not include access_token.")

        return _CachedToken(
            access_token=access_token,
            expires_at=time.time() + expires_in - TOKEN_REFRESH_SAFETY_SEC,
        )

    def _get_token(self, force_refresh: bool = False) -> str:
        cached = self._token
        if (not force_refresh
                and cached is not None
                and cached.expires_at > time.time()):
            return cached.access_token

        self._token = self._fetch_app_token()
        return self._token.access_token

    # -- Search ------------------------------------------------------------

    def search_tracks(
        self,
        query: str,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[SpotifyTrack]:
        """Run a Spotify Search for tracks and return normalised results.

        ``offset`` lets callers paginate past Spotify's relevance-sorted top
        hits, which is how Auralis surfaces a varied set of tracks for the
        same mood instead of always returning the same six results.
        """

        if not query or not query.strip():
            return []

        capped_limit  = max(1, min(int(limit), MAX_LIMIT))
        # Spotify's docs cap offset+limit at 1000 for the search endpoint.
        capped_offset = max(0, min(int(offset), 1000 - capped_limit))

        market = os.environ.get("SPOTIFY_MARKET", "US")
        params = {
            "q":      query.strip(),
            "type":   "track",
            "limit":  capped_limit,
            "offset": capped_offset,
            "market": market,
        }

        # Try once, then once more after forcing a token refresh in case the
        # cached one was revoked (e.g. after rotating the client secret).
        for attempt in (0, 1):
            token = self._get_token(force_refresh=attempt == 1)
            try:
                response = requests.get(
                    SPOTIFY_SEARCH_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    params=params,
                    timeout=self.request_timeout_sec,
                )
            except requests.RequestException as exc:
                raise SpotifyError(f"Could not reach Spotify search endpoint: {exc}") from exc

            if response.status_code == 401 and attempt == 0:
                # Token expired or revoked. Loop will refresh.
                continue

            if response.status_code != 200:
                raise SpotifyError(
                    f"Spotify search failed ({response.status_code}): {response.text}"
                )

            return self._parse_tracks(response.json())

        # Defensive -- the loop always returns or raises above.
        return []

    def find_artist(self, name: str) -> Optional[dict]:
        """Resolve an artist *name* to ``{id, name, genres}`` via Spotify search.

        We need the canonical Spotify artist ID — not just genres — so that
        downstream calls like ``get_artist_top_tracks`` can pull verified
        tracks for the correct artist. Searching ``artist:"Dave"`` via the
        track endpoint is ambiguous (Dave Matthews, Dave Brubeck, Dave the
        UK rapper all match), but the ``type=artist`` search ranks by
        Spotify's own popularity heuristic, which is reliably correct for
        the prominent artist most users mean by a short name.

        Cached so we don't pay the round-trip for repeated mentions in the
        same chat session.
        """
        if not name or not name.strip():
            return None
        key = name.strip().lower()
        cached = SpotifyClient._artist_genre_cache.get(key)
        if isinstance(cached, dict):
            return cached
        if isinstance(cached, list):
            # Legacy cache entry from before this method existed — re-resolve.
            pass

        try:
            for attempt in (0, 1):
                token = self._get_token(force_refresh=attempt == 1)
                response = requests.get(
                    SPOTIFY_SEARCH_URL,
                    headers={"Authorization": f"Bearer {token}"},
                    params={"q": name.strip(), "type": "artist", "limit": 1},
                    timeout=self.request_timeout_sec,
                )
                if response.status_code == 401 and attempt == 0:
                    continue
                if response.status_code != 200:
                    SpotifyClient._artist_genre_cache[key] = None
                    return None
                items = ((response.json().get("artists") or {}).get("items") or [])
                if not items:
                    SpotifyClient._artist_genre_cache[key] = None
                    return None
                a = items[0]
                resolved = {
                    "id":     a.get("id") or "",
                    "name":   a.get("name") or name.strip(),
                    "genres": [g for g in (a.get("genres") or []) if isinstance(g, str)],
                }
                SpotifyClient._artist_genre_cache[key] = resolved
                return resolved
        except requests.RequestException:
            return None
        return None

    def get_artist_genres(self, name: str) -> list[str]:
        """Backwards-compatible thin wrapper — returns just the genres list."""
        info = self.find_artist(name)
        return info["genres"] if info else []

    def search_artist_tracks_by_name(
        self,
        name: str,
        limit: int = 12,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[SpotifyTrack]:
        """Spotify-search-based fallback when ``/artists/{id}/top-tracks`` is
        gated by Spotify's dev-mode policy (returns 403 Forbidden).

        We can't call the artist endpoint directly, but we can search
        ``q=artist:"<name>"&type=track`` — Spotify ranks results by track
        popularity, so the most-played artist with that name dominates the
        first page. Post-filter to tracks whose **primary** artist name
        matches the requested name (case-insensitive) so we don't surface
        feature credits or songs by other artists who happen to also be
        named Dave / J. Cole / etc.

        Returns up to ``limit`` tracks. Empty list on any failure.
        """
        if not name or not name.strip():
            return []

        name_lower = name.strip().lower()
        excluded = exclude_ids or set()

        # Over-fetch since we'll filter aggressively below.
        try:
            raw_tracks = self.paginated_search_tracks(
                queries=[f'artist:"{name}"'],
                total=min(limit * 3, MAX_TOTAL_RESULTS),
                exclude_ids=excluded,
            )
        except Exception:
            return []

        matching: list[SpotifyTrack] = []
        for t in raw_tracks:
            if not t.artists:
                continue
            primary = (t.artists[0] or "").lower().strip()
            # Exact match OR the requested name is a strict prefix of the
            # primary artist name (handles "Dave" vs "Dave Matthews Band"
            # being preserved as separate matches — only the exact-name
            # primary qualifies).
            if primary == name_lower:
                matching.append(t)
            if len(matching) >= limit:
                break
        return matching

    def get_artist_album_tracks(
        self,
        artist_id: str,
        max_tracks: int = 30,
        market: Optional[str] = None,
    ) -> list[SpotifyTrack]:
        """Return tracks from an artist's albums (deep catalogue access).

        Spotify's ``/artists/{id}/top-tracks`` only returns ~10 tracks, and
        engaged listeners often have all 10 hearted already. This walks
        the artist's album list and pulls tracks from each, giving us
        access to deep cuts — what a user typically means when they ask
        for "more from X". Caps at ``max_tracks`` so we don't burn calls
        on artists with massive discographies.
        """
        if not artist_id or not artist_id.strip():
            return []
        if market is None:
            market = os.environ.get("SPOTIFY_MARKET", "US")
        artist_id = artist_id.strip()

        # Step 1: list albums (album + single, exclude appears_on /
        # compilation since those mostly recycle the catalogue).
        albums_url = f"https://api.spotify.com/v1/artists/{artist_id}/albums"
        album_ids: list[str] = []
        try:
            for attempt in (0, 1):
                token = self._get_token(force_refresh=attempt == 1)
                response = requests.get(
                    albums_url,
                    headers={"Authorization": f"Bearer {token}"},
                    # Spotify's dev-mode tightening rejects limit > 10 on
                    # the same endpoints that affected /search earlier.
                    # Stay at 10 to avoid the 400 "Invalid limit" response.
                    params={"include_groups": "album,single", "limit": 10, "market": market},
                    timeout=self.request_timeout_sec,
                )
                if response.status_code == 401 and attempt == 0:
                    continue
                if response.status_code != 200:
                    print(
                        f"[spotify] albums {artist_id} non-200: "
                        f"status={response.status_code} body={response.text[:200]!r}"
                    )
                    return []
                for it in response.json().get("items") or []:
                    aid = (it or {}).get("id")
                    if aid:
                        album_ids.append(aid)
                break
        except requests.RequestException as exc:
            print(f"[spotify] albums {artist_id} request error: {exc}")
            return []
        if not album_ids:
            print(f"[spotify] albums {artist_id} returned 0 album IDs")
            return []

        # Step 2: for each album, pull its tracks until we hit max_tracks.
        results: list[SpotifyTrack] = []
        seen_ids: set[str] = set()
        for aid in album_ids:
            if len(results) >= max_tracks:
                break
            tracks_url = f"https://api.spotify.com/v1/albums/{aid}/tracks"
            try:
                for attempt in (0, 1):
                    token = self._get_token(force_refresh=attempt == 1)
                    response = requests.get(
                        tracks_url,
                        headers={"Authorization": f"Bearer {token}"},
                        params={"limit": 50, "market": market},
                        timeout=self.request_timeout_sec,
                    )
                    if response.status_code == 401 and attempt == 0:
                        continue
                    if response.status_code != 200:
                        break
                    for t in response.json().get("items") or []:
                        if not isinstance(t, dict):
                            continue
                        tid = t.get("id")
                        if not tid or tid in seen_ids:
                            continue
                        seen_ids.add(tid)
                        # Album-tracks payload omits ``album`` so synthesise one
                        # by reusing the parent album record (we don't have
                        # the cover art URL here — fetch via track endpoint
                        # instead via the search _parse_tracks helper). To keep
                        # things simple, do a fresh lookup with the track ID
                        # via search to get the full payload (with album art).
                        results.append(SpotifyTrack(
                            spotify_id=tid,
                            name=t.get("name") or "",
                            artists=[a.get("name") for a in (t.get("artists") or []) if a.get("name")],
                            album="",
                            album_art_url=None,  # fetched lazily below
                            preview_url=t.get("preview_url"),
                            external_url=(t.get("external_urls") or {}).get("spotify", ""),
                            duration_ms=int(t.get("duration_ms") or 0),
                            explicit=bool(t.get("explicit", False)),
                        ))
                        if len(results) >= max_tracks:
                            break
                    break
            except requests.RequestException:
                continue

        # Step 3: hydrate album art — without it, the chat cards render as
        # a "♪" fallback. Spotify's /tracks?ids=... endpoint returns up to
        # 50 full track records (including album.images) in a single call.
        if results:
            try:
                ids_csv = ",".join(t.spotify_id for t in results[:50])
                token = self._get_token()
                hydrate = requests.get(
                    "https://api.spotify.com/v1/tracks",
                    headers={"Authorization": f"Bearer {token}"},
                    params={"ids": ids_csv, "market": market},
                    timeout=self.request_timeout_sec,
                )
                if hydrate.status_code == 200:
                    by_id = {
                        (item or {}).get("id"): item
                        for item in (hydrate.json().get("tracks") or [])
                        if item
                    }
                    for tr in results:
                        full = by_id.get(tr.spotify_id)
                        if not full:
                            continue
                        album = full.get("album") or {}
                        images = album.get("images") or []
                        if images:
                            mid = images[1] if len(images) > 1 else images[0]
                            tr.album_art_url = mid.get("url")
                        tr.album = album.get("name") or tr.album
            except requests.RequestException:
                pass

        return results

    def get_artist_top_tracks(
        self,
        artist_id: str,
        market: Optional[str] = None,
    ) -> list[SpotifyTrack]:
        """Return Spotify's chosen "top tracks" for an artist by ID.

        This is the only reliable way to get a *specific* artist's actual
        catalogue. The track-search endpoint is keyword-fuzzy and will
        merge results across same-name artists (Dave Matthews vs Dave the
        UK rapper), whereas this endpoint is anchored on the artist's
        canonical Spotify ID so the user gets exactly who they asked for.

        Returns up to 10 tracks. Empty list on any failure.
        """
        if not artist_id or not artist_id.strip():
            return []
        if market is None:
            market = os.environ.get("SPOTIFY_MARKET", "US")

        url = f"https://api.spotify.com/v1/artists/{artist_id.strip()}/top-tracks"
        try:
            for attempt in (0, 1):
                token = self._get_token(force_refresh=attempt == 1)
                response = requests.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                    params={"market": market},
                    timeout=self.request_timeout_sec,
                )
                if response.status_code == 401 and attempt == 0:
                    continue
                if response.status_code != 200:
                    print(
                        f"[spotify] top-tracks {artist_id} non-200: "
                        f"status={response.status_code} body={response.text[:200]!r}"
                    )
                    return []
                items = response.json().get("tracks") or []
                if not items:
                    print(f"[spotify] top-tracks {artist_id} returned 200 with empty tracks list")
                payload = {"tracks": {"items": items}}
                return self._parse_tracks(payload)
        except requests.RequestException as exc:
            print(f"[spotify] top-tracks {artist_id} request error: {exc}")
            return []
        return []

    def paginated_search_tracks(
        self,
        queries: list[str],
        total: int,
        offsets: Optional[list[int]] = None,
        exclude_ids: Optional[set[str]] = None,
        shuffle_seed: Optional[int] = None,
    ) -> list[SpotifyTrack]:
        """Run multiple search calls and stitch the results together.

        Spotify's search endpoint silently caps us at 10 results per call for
        Client-Credentials tokens (it returns 400 "Invalid limit" otherwise),
        so any UI that wants 20-30 mood-matched tracks has to fan out.

        We round-robin across the supplied ``queries`` and ``offsets`` so the
        returned list draws from multiple Spotify relevance pages instead of
        always returning the same most-popular tracks. Duplicate spotify_ids
        are dropped, and we stop once ``total`` unique tracks have been
        collected (or when we run out of fan-out combinations).

        ``exclude_ids`` lets the caller filter out tracks the user has
        already hearted/seen so successive calls surface fresh material.
        ``shuffle_seed`` randomises the (offset, query) traversal order so
        repeated calls with the same inputs produce visibly different
        lineups instead of always returning the same first N hits.
        """

        import random as _random

        target = max(1, min(int(total), MAX_TOTAL_RESULTS))
        non_empty_queries = [q for q in (queries or []) if q and q.strip()]
        if not non_empty_queries:
            return []

        # Default offsets walk past Spotify's most-popular tail so repeated
        # calls don't surface the same five chart-toppers every time.
        if offsets is None:
            offsets = [0, 10, 20, 40, 80, 150, 250]

        # Shuffle the (offset, query) traversal order so two presses with the
        # same inputs don't return the same first-N tracks. Falling back to a
        # fresh OS-level seed when the caller doesn't supply one keeps the
        # randomness genuinely fresh per call.
        rng = _random.Random(shuffle_seed)
        shuffled_offsets = list(offsets)
        rng.shuffle(shuffled_offsets)
        shuffled_queries = list(non_empty_queries)
        rng.shuffle(shuffled_queries)

        excluded = set(exclude_ids or [])
        seen_ids:   set[str] = set()
        # Track names also need to be deduped: Spotify assigns different IDs
        # to the single, album, deluxe, and regional masters of the same
        # recording (e.g. "Sushi and Chill - Drey Karper" appearing twice),
        # which made the playlist feel repetitive. We additionally dedupe by
        # a normalised "title - primary_artist" key so a 30-track playlist
        # contains 30 *distinct* songs, not 30 distinct Spotify IDs.
        seen_titles: set[str] = set()
        collected: list[SpotifyTrack] = []

        def _title_key(t: "SpotifyTrack") -> str:
            primary = (t.artists[0] if t.artists else "").strip().lower()
            # Strip Spotify's parenthetical version tags so "Mad World" and
            # "Mad World (2003 Remaster)" collapse to one entry.
            name = (t.name or "").strip().lower()
            for marker in (" - ", " (", " feat.", " feat ", " ft."):
                idx = name.find(marker)
                if idx > 0:
                    name = name[:idx]
                    break
            return f"{name.strip()}|{primary}"

        # Fan out: every (query, offset) pairing until we hit ``target`` or
        # exhaust the grid. Pulling 10 per call keeps every individual request
        # well inside Spotify's documented limits.
        for offset in shuffled_offsets:
            for query in shuffled_queries:
                if len(collected) >= target:
                    return collected[:target]
                try:
                    page = self.search_tracks(query=query, limit=MAX_LIMIT, offset=offset)
                except SpotifyError:
                    # One bad query (e.g. malformed artist clause) shouldn't
                    # take the whole playlist down. Skip and keep going.
                    continue
                for track in page:
                    if not track.spotify_id or track.spotify_id in seen_ids:
                        continue
                    if track.spotify_id in excluded:
                        continue
                    # Drop royalty-free / stock-music / generic instrumental
                    # filler before it ever reaches the recommendation pool.
                    # Listeners want real songs; production-music tracks like
                    # "Uplifting Pop" by AShamaluevMusic just dilute the rail.
                    if is_stock_or_instrumental(track):
                        continue
                    title_key = _title_key(track)
                    if title_key in seen_titles:
                        continue
                    seen_ids.add(track.spotify_id)
                    seen_titles.add(title_key)
                    collected.append(track)
                    if len(collected) >= target:
                        return collected[:target]

        return collected[:target]

    @staticmethod
    def _parse_tracks(payload: dict) -> list[SpotifyTrack]:
        items = (payload.get("tracks") or {}).get("items") or []
        results: list[SpotifyTrack] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            album = item.get("album") or {}
            images = album.get("images") or []
            # Spotify returns images in descending size; the second one is a
            # good middle-of-the-road thumbnail for cards.
            art_url = None
            if images:
                mid = images[1] if len(images) > 1 else images[0]
                art_url = mid.get("url")
            artists = [
                a.get("name") for a in (item.get("artists") or []) if isinstance(a, dict) and a.get("name")
            ]
            results.append(
                SpotifyTrack(
                    spotify_id=item.get("id") or "",
                    name=item.get("name") or "",
                    artists=artists,
                    album=album.get("name") or "",
                    album_art_url=art_url,
                    preview_url=item.get("preview_url"),
                    external_url=(item.get("external_urls") or {}).get("spotify", ""),
                    duration_ms=int(item.get("duration_ms") or 0),
                    explicit=bool(item.get("explicit", False)),
                )
            )
        return results
