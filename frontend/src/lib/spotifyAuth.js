/**
 * Client-side PKCE OAuth 2.1 flow for Spotify.
 *
 * Why PKCE in the browser
 * -----------------------
 * Spotify's PKCE flow is the only auth model that doesn't need a client
 * secret to exchange the authorization code, which means we can keep the
 * secret server-side (Fly.io) for the app-level Search calls and run user
 * auth entirely in the browser. The user's access token never touches our
 * backend, so we never log it, store it, or expose it through our logs.
 *
 * Token storage
 * -------------
 * - access_token + refresh_token + expiry are stored in localStorage so the
 *   user stays logged in across page reloads.
 * - The PKCE code_verifier lives in sessionStorage during the authorize
 *   redirect, then is consumed and deleted on /callback.
 *
 * Scopes
 * ------
 * - user-library-modify    : "+ Save" button writes to the user's library
 * - playlist-modify-private: "Push to Spotify" creates a private playlist
 * - playlist-modify-public : Spotify's docs say only -private is required
 *   to add tracks to a private playlist, but as of early 2026 the
 *   POST /playlists/{id}/tracks endpoint started returning 403 Forbidden
 *   on Development-mode apps unless -public is also requested. Asking for
 *   both unblocks the add-tracks call without changing user-visible
 *   behaviour (created playlists remain private).
 * - user-read-email        : show "Connected as you@example.com"
 * - user-top-read          : pull top artists for For You + Playlist blend
 */

// Public-by-design. Spotify's docs explicitly say the Client ID can be
// shipped to browsers. Override via VITE_SPOTIFY_CLIENT_ID in env if you
// want to point a deploy at a different Spotify app.
export const SPOTIFY_CLIENT_ID =
  import.meta.env.VITE_SPOTIFY_CLIENT_ID ||
  'f70110980a5d4bfa8c04b4da79fb2d7c'

export const SPOTIFY_SCOPES = [
  'user-library-modify',
  'playlist-modify-private',
  'playlist-modify-public',
  'user-read-email',
  'user-top-read',
].join(' ')

const AUTHORIZE_URL = 'https://accounts.spotify.com/authorize'
const TOKEN_URL     = 'https://accounts.spotify.com/api/token'

const LS_TOKEN_KEY    = 'auralis.spotify.tokens'
const SS_VERIFIER_KEY = 'auralis.spotify.pkce_verifier'
const SS_STATE_KEY    = 'auralis.spotify.oauth_state'

// Token refresh runs this many seconds before the real expiry so a slow
// network call can't straddle the boundary.
const REFRESH_SAFETY_SEC = 60


// ── PKCE primitives ─────────────────────────────────────────────────────────

/**
 * Returns the redirect URI to register in the Spotify dashboard for the
 * current origin. We use 127.0.0.1 (not localhost) for dev because
 * Spotify's dashboard accepts the loopback IP literal over HTTP but not
 * the "localhost" hostname.
 */
export function getRedirectUri() {
  if (typeof window === 'undefined') return ''
  const origin = window.location.origin
  // Normalise localhost → 127.0.0.1 so the registered URI matches what the
  // browser actually sends when developers type either address.
  const normalised = origin.replace('://localhost', '://127.0.0.1')
  return `${normalised}/callback`
}

function randomString(length) {
  const bytes = new Uint8Array(length)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => (b % 36).toString(36)).join('')
}

function base64UrlEncode(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')
}

async function sha256(plain) {
  const buffer = new TextEncoder().encode(plain)
  return crypto.subtle.digest('SHA-256', buffer)
}


// ── Authorize redirect ──────────────────────────────────────────────────────

export async function beginLogin() {
  const verifier  = randomString(64)
  const challenge = base64UrlEncode(await sha256(verifier))
  const state     = randomString(16)

  sessionStorage.setItem(SS_VERIFIER_KEY, verifier)
  sessionStorage.setItem(SS_STATE_KEY, state)

  const params = new URLSearchParams({
    response_type:         'code',
    client_id:             SPOTIFY_CLIENT_ID,
    scope:                 SPOTIFY_SCOPES,
    redirect_uri:          getRedirectUri(),
    state,
    code_challenge_method: 'S256',
    code_challenge:        challenge,
  })

  window.location.assign(`${AUTHORIZE_URL}?${params.toString()}`)
}


// ── Code exchange (called from /callback) ───────────────────────────────────

export async function completeLogin({ code, state }) {
  const expectedState = sessionStorage.getItem(SS_STATE_KEY)
  const verifier      = sessionStorage.getItem(SS_VERIFIER_KEY)

  if (!verifier) {
    throw new Error('Missing PKCE verifier. Please start login again.')
  }
  if (!state || state !== expectedState) {
    throw new Error('OAuth state mismatch. Possible CSRF — please try again.')
  }

  const body = new URLSearchParams({
    grant_type:    'authorization_code',
    code,
    redirect_uri:  getRedirectUri(),
    client_id:     SPOTIFY_CLIENT_ID,
    code_verifier: verifier,
  })

  const res = await fetch(TOKEN_URL, {
    method:  'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })

  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Spotify token exchange failed (${res.status}): ${detail}`)
  }

  const data = await res.json()
  saveTokens(data)

  // One-time use; clear immediately so a refresh of /callback can't replay.
  sessionStorage.removeItem(SS_VERIFIER_KEY)
  sessionStorage.removeItem(SS_STATE_KEY)
}


// ── Refresh ─────────────────────────────────────────────────────────────────

async function refreshTokens(refreshToken) {
  const body = new URLSearchParams({
    grant_type:    'refresh_token',
    refresh_token: refreshToken,
    client_id:     SPOTIFY_CLIENT_ID,
  })

  const res = await fetch(TOKEN_URL, {
    method:  'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })

  if (!res.ok) {
    // Refresh token revoked or otherwise invalid. Force a fresh login.
    logout()
    throw new Error(`Token refresh failed (${res.status}). Please reconnect Spotify.`)
  }

  const data = await res.json()
  // Spotify usually does NOT rotate the refresh_token on each refresh, so
  // keep the old one if the response omits a new value.
  data.refresh_token = data.refresh_token || refreshToken
  saveTokens(data)
}


// ── Storage helpers ─────────────────────────────────────────────────────────

function saveTokens(data) {
  const tokens = {
    access_token:  data.access_token,
    refresh_token: data.refresh_token,
    token_type:    data.token_type || 'Bearer',
    scope:         data.scope || SPOTIFY_SCOPES,
    expires_at:    Date.now() + (data.expires_in || 3600) * 1000 - REFRESH_SAFETY_SEC * 1000,
  }
  localStorage.setItem(LS_TOKEN_KEY, JSON.stringify(tokens))
}

export function loadTokens() {
  try {
    const raw = localStorage.getItem(LS_TOKEN_KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function logout() {
  localStorage.removeItem(LS_TOKEN_KEY)
  sessionStorage.removeItem(SS_VERIFIER_KEY)
  sessionStorage.removeItem(SS_STATE_KEY)
}


// ── Public access-token getter (auto-refreshes when needed) ─────────────────

/**
 * Returns a valid Bearer token, refreshing it transparently if it's near
 * expiry. Returns null if the user is not logged in or refresh fails.
 */
export async function getAccessToken() {
  const tokens = loadTokens()
  if (!tokens) return null

  if (Date.now() >= tokens.expires_at) {
    if (!tokens.refresh_token) {
      logout()
      return null
    }
    try {
      await refreshTokens(tokens.refresh_token)
    } catch {
      return null
    }
    return loadTokens()?.access_token || null
  }

  return tokens.access_token
}

export function isLoggedIn() {
  return loadTokens() !== null
}
