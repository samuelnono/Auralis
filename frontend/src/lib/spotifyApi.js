/**
 * Thin wrappers around Spotify's user-scoped Web API endpoints.
 *
 * These functions all rely on the user being logged in via the PKCE flow in
 * `spotifyAuth.js`. They call `getAccessToken()` on every request so the token
 * is auto-refreshed transparently when it's near expiry — callers never need
 * to think about the token lifecycle.
 *
 * Why this lives client-side
 * --------------------------
 * Auralis's Phase 2 design keeps user-scoped Spotify calls in the browser so
 * the access token never traverses our backend. Lower attack surface, simpler
 * infra, and the user can revoke our app's access at any time from
 * https://www.spotify.com/account/apps without us holding any state.
 *
 * Error handling
 * --------------
 * Each function throws on non-2xx responses with a readable message. Spotify's
 * error bodies are usually `{ error: { status, message } }`, so we surface
 * `error.message` when present and fall back to the raw response text.
 */

import { getAccessToken } from './spotifyAuth'

const API_BASE = 'https://api.spotify.com/v1'


// ── Internal request helper ─────────────────────────────────────────────────

async function spotifyFetch(path, { method = 'GET', body, headers = {} } = {}) {
  const token = await getAccessToken()
  if (!token) {
    throw new Error('Not connected to Spotify. Please log in first.')
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      // Only set Content-Type when there's a JSON body — Spotify is picky about
      // PUT requests that include the header but no body (returns 400).
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

  // 204 No Content is success for save/add-to-playlist. No body to parse.
  if (res.status === 204) return null

  const text = await res.text()
  let data = null
  try { data = text ? JSON.parse(text) : null } catch { /* keep text */ }

  if (!res.ok) {
    const message = data?.error?.message || text || `HTTP ${res.status}`
    throw new Error(`Spotify API ${res.status}: ${message}`)
  }

  return data
}


// ── User profile ────────────────────────────────────────────────────────────

/**
 * Returns the logged-in user's profile. We use this for two things:
 *   1. The user's Spotify ID, which is required to create playlists.
 *   2. The display name + email shown in the "Connected as ..." UI.
 */
export async function getCurrentUser() {
  return spotifyFetch('/me')
}


// ── Listening history (top tracks / artists) ────────────────────────────────

/**
 * Returns the user's top artists. `time_range` is one of:
 *   - 'short_term'  ≈ last 4 weeks
 *   - 'medium_term' ≈ last 6 months  (default — matches Spotify's UI default)
 *   - 'long_term'   ≈ all time
 *
 * Used by the For You page to drive recommendations off the listener's actual
 * listening history rather than only the locally-rated tracks.
 */
export async function getTopArtists({ timeRange = 'medium_term', limit = 10 } = {}) {
  const params = new URLSearchParams({ time_range: timeRange, limit: String(limit) })
  return spotifyFetch(`/me/top/artists?${params.toString()}`)
}

/**
 * Returns the user's top tracks. Same time_range options as getTopArtists.
 */
export async function getTopTracks({ timeRange = 'medium_term', limit = 10 } = {}) {
  const params = new URLSearchParams({ time_range: timeRange, limit: String(limit) })
  return spotifyFetch(`/me/top/tracks?${params.toString()}`)
}


// ── Library: save tracks ────────────────────────────────────────────────────

/**
 * Saves one or more tracks to the user's "Liked Songs" library.
 * Idempotent — calling twice with the same ID is a no-op on Spotify's side.
 */
export async function saveTracks(trackIds) {
  const ids = (Array.isArray(trackIds) ? trackIds : [trackIds]).filter(Boolean)
  if (ids.length === 0) return null
  // Spotify caps this at 50 IDs per call. We never exceed that in the UI but
  // chunk defensively in case a future caller passes a long array.
  for (let i = 0; i < ids.length; i += 50) {
    const chunk = ids.slice(i, i + 50)
    await spotifyFetch('/me/tracks', { method: 'PUT', body: { ids: chunk } })
  }
  return { saved: ids.length }
}

/**
 * Removes one or more tracks from the user's "Liked Songs" library.
 */
export async function removeSavedTracks(trackIds) {
  const ids = (Array.isArray(trackIds) ? trackIds : [trackIds]).filter(Boolean)
  if (ids.length === 0) return null
  for (let i = 0; i < ids.length; i += 50) {
    const chunk = ids.slice(i, i + 50)
    await spotifyFetch('/me/tracks', { method: 'DELETE', body: { ids: chunk } })
  }
  return { removed: ids.length }
}

/**
 * Checks whether the given track IDs are already saved. Returns an array of
 * booleans in the same order as the input.
 */
export async function checkSavedTracks(trackIds) {
  const ids = (Array.isArray(trackIds) ? trackIds : [trackIds]).filter(Boolean)
  if (ids.length === 0) return []
  const params = new URLSearchParams({ ids: ids.join(',') })
  return spotifyFetch(`/me/tracks/contains?${params.toString()}`)
}


// ── Playlists ──────────────────────────────────────────────────────────────

/**
 * Creates a private playlist on the user's account and returns the new
 * playlist object (we mostly care about `id` and `external_urls.spotify`).
 *
 * Defaults to private + non-collaborative so the user has to opt into sharing.
 *
 * Endpoint choice: we hit ``POST /me/playlists`` rather than the equivalent
 * ``POST /users/{user_id}/playlists``. Both are documented as equivalent for
 * creating a playlist on the authenticated user's account, but Spotify
 * Development-mode apps occasionally 403 the user-ID-in-URL form even when
 * the user is allowlisted and the token carries playlist-modify-private. The
 * /me/ form resolves the owner implicitly from the bearer token and routes
 * through a slightly different code path on Spotify's side, which sidesteps
 * that quirk. The ``userId`` parameter is still accepted for backwards
 * compatibility but is no longer used.
 */
export async function createPlaylist(userId, { name, description = '', isPublic = false } = {}) {
  if (!name) throw new Error('createPlaylist requires a name.')
  return spotifyFetch('/me/playlists', {
    method: 'POST',
    body: { name, description, public: isPublic },
  })
}

/**
 * Adds tracks to an existing playlist. Spotify needs full URIs
 * (`spotify:track:<id>`), so we accept either raw IDs or full URIs and
 * normalise.
 */
export async function addTracksToPlaylist(playlistId, trackIdsOrUris) {
  if (!playlistId) throw new Error('addTracksToPlaylist requires a playlistId.')
  const uris = (Array.isArray(trackIdsOrUris) ? trackIdsOrUris : [trackIdsOrUris])
    .filter(Boolean)
    .map((v) => (v.startsWith('spotify:track:') ? v : `spotify:track:${v}`))
  if (uris.length === 0) return null
  // Spotify caps this at 100 URIs per call.
  for (let i = 0; i < uris.length; i += 100) {
    const chunk = uris.slice(i, i + 100)
    await spotifyFetch(`/playlists/${encodeURIComponent(playlistId)}/tracks`, {
      method: 'POST',
      body: { uris: chunk },
    })
  }
  return { added: uris.length }
}

/**
 * One-shot convenience: create a playlist for the current user and fill it
 * with the given track IDs. Returns the created playlist object so callers
 * can link the user straight to it on Spotify.
 */
export async function createPlaylistWithTracks({ name, description, trackIds, isPublic = false }) {
  const me = await getCurrentUser()
  const playlist = await createPlaylist(me.id, { name, description, isPublic })
  if (trackIds && trackIds.length > 0) {
    await addTracksToPlaylist(playlist.id, trackIds)
  }
  return playlist
}
