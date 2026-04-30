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
from typing import Optional

import requests


SPOTIFY_TOKEN_URL  = "https://accounts.spotify.com/api/token"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"

# Spotify Search caps the page size at 50; the docs say 10 is the default.
# We keep the default low because the UI surfaces a handful of tracks per mood.
DEFAULT_LIMIT = 10
MAX_LIMIT     = 50

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

    def __post_init__(self) -> None:
        # Allow zero-arg construction in tests by reading env at call time too,
        # but resolving here gives us an early failure if both are missing.
        self.client_id     = self.client_id     or os.environ.get("SPOTIFY_CLIENT_ID")
        self.client_secret = self.client_secret or os.environ.get("SPOTIFY_CLIENT_SECRET")

    # ── Auth ──────────────────────────────────────────────────────────────

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

    # ── Search ────────────────────────────────────────────────────────────

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

        # No `market=from_token` here — that value requires a user-scoped token,
        # not the app-level Client Credentials token we're using. Pinning a
        # specific market keeps results sensible (and Spotify is happier with
        # an explicit ISO country code than nothing at all). Override via env
        # if you need a different region for testing.
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

        # Defensive — the loop always returns or raises above.
        return []

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
