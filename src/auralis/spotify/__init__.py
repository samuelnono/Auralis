"""Spotify integration for Auralis.

Phase 1 exposes app-level (Client Credentials) Search so we can surface real
Spotify track suggestions for any analysed mood. Phase 2 will add user-scope
PKCE auth so listeners can save those suggestions back to their library.
"""

from .client import SpotifyClient, SpotifyTrack, SpotifyError
from .mood_map import mood_to_query

__all__ = ["SpotifyClient", "SpotifyTrack", "SpotifyError", "mood_to_query"]
