/**
 * Playlist page.
 *
 * Two sources are now available:
 *   1. Spotify (default) — pulls real tracks via the backend's
 *      /spotify/search-by-mood endpoint, renders proper cards with album art,
 *      and lets logged-in users push the result straight to their Spotify
 *      account as a private playlist.
 *   2. Local index — the original research view that ranks tracks from the
 *      training dataset by emotion match. Useful for explaining the model,
 *      less useful as a primary listening surface since dataset tracks aren't
 *      user-playable in-app.
 */

import { useEffect, useState } from 'react'
import axios from 'axios'
import { API } from '../config'
import { useSpotifyAuth } from '../hooks/useSpotifyAuth'
import {
  createPlaylistWithTracks,
  createPlaylist,
  getCurrentUser,
  getTopArtists,
  saveTracks,
} from '../lib/spotifyApi'

const EMOTION_EMOJIS = { calm: '🌊', energetic: '⚡', happy: '☀️', sad: '🌧️' }
const MOOD_DESCRIPTIONS = {
  calm: 'Low tempo, smooth textures, gentle energy',
  energetic: 'High BPM, dynamic range, driving rhythms',
  happy: 'Bright timbre, uplifting spectral centroid',
  sad: 'Low brightness, minor spectral patterns',
}

// Spotify's Feb 2026 dev-mode tightening blocks PUT /me/tracks (the
// Liked-Songs write endpoint) until an app is approved for Extended
// Quota / production access. Until that approval lands, hide the
// "♥ Save all to Liked Songs" button — the underlying handler and
// state are intentionally left in place so flipping this flag back
// to true is a one-line change with no other code edits required.
const SHOW_SAVE_TO_LIKED_BUTTON = false

export default function Playlist({ profile }) {
  const [source, setSource] = useState('spotify')   // 'spotify' | 'local'
  const [mode, setMode]     = useState('emotion')   // 'emotion' | 'profile'
  const [targetEmotion, setTargetEmotion] = useState('calm')
  // Default to a fuller playlist; the slider can scale up to 30 (Spotify
  // search returns at most 50 per call so we stay well under that).
  const [length, setLength] = useState(12)

  // Spotify-mode state
  const [spotifyTracks, setSpotifyTracks]   = useState([])
  const [spotifyMeta, setSpotifyMeta]       = useState(null)
  const [pushedPlaylist, setPushedPlaylist] = useState(null)
  const [pushing, setPushing]               = useState(false)
  // "Save all to Liked Songs" state — independent from pushing because the
  // user might do one, the other, or both. ``savedCount`` is null when no
  // save has happened yet, an integer once a save completes. ``saveBlocked``
  // is set when Spotify 403s the PUT /me/tracks call (Dev-Mode quota block);
  // we surface a soft explainer card pointing at the "As playlist" path
  // instead of a raw API error string.
  const [saving,      setSaving]      = useState(false)
  const [savedCount,  setSavedCount]  = useState(null)
  const [saveBlocked, setSaveBlocked] = useState(false)

  // Local-mode state
  const [playlist, setPlaylist] = useState([])

  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  // Spotify-seed state — pre-fetched on login so the Generate button doesn't
  // pay a round-trip every time. We forward `seedArtists` and `seedGenres`
  // to the backend, which mixes them into the Search-query pool so the
  // playlist blends the chosen mood with the user's actual taste.
  const [seedArtists, setSeedArtists] = useState([])
  const [seedGenres,  setSeedGenres]  = useState([])

  const spotifyAuth = useSpotifyAuth()

  // Pull top artists once whenever the user becomes logged in. We keep the
  // list short (top 8) so the artist:"..." clauses we send to Spotify Search
  // stay well under any URL-length / query-complexity limits.
  useEffect(() => {
    let cancelled = false
    if (!spotifyAuth.isLoggedIn) {
      setSeedArtists([])
      setSeedGenres([])
      return
    }
    ;(async () => {
      try {
        const data = await getTopArtists({ timeRange: 'medium_term', limit: 8 })
        if (cancelled) return
        const items = data?.items || []
        setSeedArtists(items.map((a) => a?.name).filter(Boolean))
        // Spotify returns a `genres` array on each artist; flatten + dedupe
        // for a small set of taste-flavoured genre tags.
        const genres = []
        const seen = new Set()
        for (const a of items) {
          for (const g of (a?.genres || [])) {
            if (!seen.has(g)) {
              seen.add(g)
              genres.push(g)
            }
          }
          if (genres.length >= 6) break
        }
        setSeedGenres(genres.slice(0, 6))
      } catch (_) {
        // Non-fatal: playlist generation still works without seeds.
        if (!cancelled) {
          setSeedArtists([])
          setSeedGenres([])
        }
      }
    })()
    return () => { cancelled = true }
  }, [spotifyAuth.isLoggedIn])

  const effectiveEmotion = mode === 'profile' && profile?.has_signal
    ? profile.dominant_emotion
    : targetEmotion

  const generate = async () => {
    setLoading(true)
    setError(null)
    setPushedPlaylist(null)
    setSavedCount(null)
    setSaveBlocked(false)

    try {
      if (source === 'spotify') {
        // Forward the user's Spotify seeds when we have them; the backend
        // mixes them into the Search-query pool so the playlist reflects
        // both the requested mood AND the listener's actual taste.
        const res = await axios.post(`${API}/spotify/search-by-mood`, {
          discrete_emotion: effectiveEmotion,
          limit: length,
          seed_artists: seedArtists,
          seed_genres:  seedGenres,
        })
        setSpotifyTracks(res.data.tracks || [])
        setSpotifyMeta({
          query:     res.data.query,
          quadrant:  res.data.quadrant,
          rationale: res.data.rationale,
          blended:   res.data.blended,
        })
        setPlaylist([])
      } else {
        const res = await axios.post(`${API}/playlist`, {
          mode,
          target_emotion: targetEmotion,
          length,
        })
        setPlaylist(res.data.playlist)
        setSpotifyTracks([])
      }
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to generate playlist')
    } finally {
      setLoading(false)
    }
  }

  const exportCSV = () => {
    window.open(`${API}/playlist/export?mode=${mode}&target_emotion=${targetEmotion}&length=${length}`)
  }

  const pushToSpotify = async () => {
    if (!spotifyAuth.isLoggedIn) {
      // Kick straight into the OAuth dance — no point making the user click
      // again on the Profile page first.
      await spotifyAuth.login()
      return
    }
    if (spotifyTracks.length === 0) return

    setPushing(true)
    setError(null)
    try {
      const dateStr = new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
      const name = `Auralis · ${effectiveEmotion} (${dateStr})`
      const description =
        `Auralis-generated playlist for a ${effectiveEmotion} mood. ` +
        `${spotifyMeta?.rationale || ''}`.trim()
      const trackIds = spotifyTracks.map((t) => t.spotify_id).filter(Boolean)
      const uriList = trackIds.map((id) => `spotify:track:${id}`).join('\n')

      // Try the full push first. createPlaylistWithTracks always succeeds at
      // creating the playlist (POST /me/playlists works on Dev-mode apps as
      // of early 2026), and only the secondary "add tracks" call may 403.
      // We catch THAT specific failure and fall back to a one-paste flow:
      // copy the URIs to clipboard and open the (empty) Spotify playlist in
      // a new tab so the user can press Ctrl+V to drop them in. The
      // playlist still ends up on their account; the only manual step is
      // the paste.
      try {
        const created = await createPlaylistWithTracks({
          name,
          description,
          trackIds,
          isPublic: false,
        })
        setPushedPlaylist({ ...created, mode: 'full' })
      } catch (innerErr) {
        const msg = innerErr?.message || ''
        const looksLikeAddTracksBlock =
          msg.includes('403') || msg.toLowerCase().includes('forbidden')

        if (!looksLikeAddTracksBlock) {
          throw innerErr
        }

        // Re-create the playlist on its own (or reuse one we already made
        // in a prior failed attempt this session) and degrade to clipboard.
        const me      = await getCurrentUser()
        const created = await createPlaylist(me.id, { name, description, isPublic: false })

        try {
          await navigator.clipboard.writeText(uriList)
        } catch {
          // clipboard write requires a user gesture in some browsers; the
          // pasteable list is also surfaced in the success card below as a
          // visible textarea so the user always has a manual fallback.
        }

        setPushedPlaylist({
          ...created,
          mode: 'clipboard',
          uriList,
          trackCount: trackIds.length,
        })
      }
    } catch (e) {
      setError(e.message || 'Failed to push playlist to Spotify.')
    } finally {
      setPushing(false)
    }
  }

  /**
   * "Save all to Liked Songs" — uses PUT /me/tracks under the hood.
   *
   * As of Spotify's Feb 2026 Developer-Mode tightening, write endpoints
   * (including /me/tracks for Liked Songs) return 403 Forbidden for apps
   * that haven't been granted Extended Quota. We catch that specific
   * failure and surface a soft, demo-friendly explanation card pointing
   * the user at the "As playlist" path instead, rather than a raw
   * "Spotify API 403: Forbidden" error. Once Auralis is approved for
   * production access, this code path will start working with no frontend
   * change needed.
   */
  const saveAllToLiked = async () => {
    if (!spotifyAuth.isLoggedIn) {
      await spotifyAuth.login()
      return
    }
    if (spotifyTracks.length === 0) return

    setSaving(true)
    setError(null)
    setSaveBlocked(false)
    try {
      const trackIds = spotifyTracks.map((t) => t.spotify_id).filter(Boolean)
      await saveTracks(trackIds)
      setSavedCount(trackIds.length)
    } catch (e) {
      const msg = e?.message || ''
      const isDevModeBlock =
        msg.includes('403') || msg.toLowerCase().includes('forbidden')
      if (isDevModeBlock) {
        setSaveBlocked(true)
      } else {
        setError(msg || 'Failed to save tracks to your Liked Songs.')
      }
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Playlist</h1>
        <p className="page-subtitle">
          Generate an emotion-aware playlist — from Spotify directly, or from your
          local research index.
        </p>
      </div>

      {/* Source toggle */}
      <div className="source-toggle">
        <button
          className={`source-toggle-btn ${source === 'spotify' ? 'active' : ''}`}
          onClick={() => { setSource('spotify'); setError(null) }}
        >
          <span className="source-toggle-dot spotify" />
          Spotify
          <span className="source-toggle-hint">live tracks · push to your account</span>
        </button>
        <button
          className={`source-toggle-btn ${source === 'local' ? 'active' : ''}`}
          onClick={() => { setSource('local'); setError(null) }}
        >
          <span className="source-toggle-dot local" />
          Local index
          <span className="source-toggle-hint">research view · MFCC similarity</span>
        </button>
      </div>

      {/* Mode selector (emotion vs profile) — applies to both sources */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        <button className={`btn ${mode === 'emotion' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('emotion')}>
          🎭 By Emotion
        </button>
        <button
          className={`btn ${mode === 'profile' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setMode('profile')}
          disabled={!profile?.has_signal}
          title={!profile?.has_signal ? 'Rate some tracks first' : ''}
        >
          ◉ My Profile
        </button>
      </div>

      {/* Mood cards (shown when emotion mode) */}
      {mode === 'emotion' && (
        <div className="mood-grid" style={{ marginBottom: 24 }}>
          {['calm', 'energetic', 'happy', 'sad'].map(emotion => (
            <div
              key={emotion}
              className={`mood-card ${emotion} ${targetEmotion === emotion ? 'selected' : ''}`}
              onClick={() => setTargetEmotion(emotion)}
            >
              <div className="mood-card-bg">{EMOTION_EMOJIS[emotion]}</div>
              <div className="mood-card-emoji">{EMOTION_EMOJIS[emotion]}</div>
              <div className="mood-card-name">{emotion}</div>
              <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.5)', marginTop: 4 }}>
                {MOOD_DESCRIPTIONS[emotion]}
              </div>
            </div>
          ))}
        </div>
      )}

      {mode === 'profile' && profile?.has_signal && (
        <div className="card" style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: 32 }}>{EMOTION_EMOJIS[profile.dominant_emotion]}</span>
            <div>
              <div style={{ fontFamily: 'Syne, sans-serif', fontWeight: 700, fontSize: 16 }}>
                Matching your profile
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Dominant taste: <span style={{ color: 'var(--accent)' }}>{profile.dominant_emotion}</span>
                {' · '}{profile.total_likes} liked tracks
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Length slider */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="slider-row">
          <div className="slider-label">
            <span>Playlist length</span>
            <span className="slider-value">{length} tracks</span>
          </div>
          <input type="range" min={3} max={30} step={1} value={length}
            onChange={e => setLength(parseInt(e.target.value))} />
        </div>
      </div>

      <button className="btn btn-primary" onClick={generate} disabled={loading}
        style={{ marginBottom: 32, width: '100%' }}>
        {loading ? '⟳ Generating...' : '⋮⋮ Generate Playlist'}
      </button>

      {loading && <div className="loading"><div className="spinner" /> Building playlist...</div>}
      {error && <div style={{ color: '#ff6b8a', fontSize: 13, marginBottom: 16 }}>{error}</div>}

      {/* ── SPOTIFY MODE RESULTS ── */}
      {source === 'spotify' && spotifyTracks.length > 0 && (
        <>
          <div className="section-header">
            <div className="section-title">
              {spotifyTracks.length} Spotify tracks · {effectiveEmotion}
              {spotifyMeta?.blended && (
                <span style={{ marginLeft: 8, fontSize: 11, color: '#1ed760' }}>
                  · blended with your taste
                </span>
              )}
            </div>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              {SHOW_SAVE_TO_LIKED_BUTTON && (
                <button
                  className="btn btn-primary"
                  onClick={saveAllToLiked}
                  disabled={saving || pushing}
                  style={{ background: '#1ed760', borderColor: '#1ed760', color: '#000', fontSize: 12 }}
                  title="Adds every track to your Spotify Liked Songs in one click. No paste needed."
                >
                  {saving
                    ? '⟳ Saving…'
                    : spotifyAuth.isLoggedIn
                      ? `♥ Save all to Liked Songs`
                      : 'Connect Spotify to save'}
                </button>
              )}
              <button
                className="btn btn-primary"
                onClick={pushToSpotify}
                disabled={pushing || saving}
                style={{ background: '#1ed760', borderColor: '#1ed760', color: '#000', fontSize: 12 }}
                title="Creates a Spotify playlist on your account; one Ctrl+V in Spotify drops the tracks in."
              >
                {pushing
                  ? '⟳ Pushing…'
                  : spotifyAuth.isLoggedIn
                    ? '↑ Push to Spotify'
                    : 'Connect Spotify to save'}
              </button>
            </div>
          </div>

          {spotifyMeta?.rationale && (
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 16, lineHeight: 1.5 }}>
              {spotifyMeta.rationale}
            </div>
          )}

          {saveBlocked && (
            <div
              className="card"
              style={{
                marginBottom: 16,
                background: 'rgba(255, 196, 86, 0.06)',
                borderColor: 'rgba(255, 196, 86, 0.25)',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 6 }}>
                Liked-Songs save is locked while Auralis is in Spotify Dev Mode
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                Spotify's Feb 2026 policy change blocks library writes for
                apps that haven't been approved for production yet. Use{' '}
                <strong>↑ As playlist</strong> instead — same tracks, one
                paste step in Spotify, and the result lives on your account
                as a private playlist. This button will start working
                automatically once the app is approved.
              </div>
            </div>
          )}

          {savedCount !== null && savedCount > 0 && (
            <div
              className="card"
              style={{
                marginBottom: 16,
                background: 'rgba(255, 107, 138, 0.08)',
                borderColor: 'rgba(255, 107, 138, 0.3)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>
                    ♥ Added {savedCount} track{savedCount === 1 ? '' : 's'} to your Liked Songs
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    Open Spotify → Liked Songs → sort by Recently Added to see them.
                  </div>
                </div>
                <a
                  className="btn btn-ghost"
                  href="https://open.spotify.com/collection/tracks"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ fontSize: 12 }}
                >
                  Open Liked Songs ↗
                </a>
              </div>
            </div>
          )}

          {pushedPlaylist && pushedPlaylist.mode !== 'clipboard' && (
            <div
              className="card"
              style={{
                marginBottom: 16,
                background: 'rgba(30, 215, 96, 0.08)',
                borderColor: 'rgba(30, 215, 96, 0.3)',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                <div>
                  <div style={{ fontWeight: 600, marginBottom: 4 }}>
                    Saved to Spotify as “{pushedPlaylist.name}”
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    Private playlist on your account
                  </div>
                </div>
                <a
                  className="btn btn-ghost"
                  href={pushedPlaylist.external_urls?.spotify}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ fontSize: 12 }}
                >
                  Open in Spotify ↗
                </a>
              </div>
            </div>
          )}

          {pushedPlaylist && pushedPlaylist.mode === 'clipboard' && (
            <div
              className="card"
              style={{
                marginBottom: 16,
                background: 'rgba(30, 215, 96, 0.08)',
                borderColor: 'rgba(30, 215, 96, 0.3)',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 6 }}>
                Created “{pushedPlaylist.name}” on your Spotify account
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 12, lineHeight: 1.5 }}>
                Spotify is currently restricting <em>add-tracks</em> calls for apps in
                Developer Mode (their Feb 2026 policy change), so the {pushedPlaylist.trackCount}-track list
                couldn’t be auto-filled. We copied the track URIs to your clipboard — open the
                playlist in Spotify, click in the track list, and press <strong>Ctrl + V</strong>
                to drop them in.
              </div>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                <a
                  className="btn btn-primary"
                  href={pushedPlaylist.external_urls?.spotify}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ background: '#1ed760', borderColor: '#1ed760', color: '#000', fontSize: 12 }}
                >
                  Open playlist in Spotify ↗
                </a>
                <button
                  className="btn btn-ghost"
                  style={{ fontSize: 12 }}
                  onClick={() => {
                    navigator.clipboard?.writeText(pushedPlaylist.uriList).catch(() => {})
                  }}
                >
                  Copy URIs again
                </button>
              </div>
              <details style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                <summary style={{ cursor: 'pointer' }}>Show pasteable track URIs</summary>
                <textarea
                  readOnly
                  value={pushedPlaylist.uriList}
                  rows={Math.min(pushedPlaylist.trackCount, 8)}
                  style={{
                    marginTop: 8,
                    width: '100%',
                    background: 'rgba(0,0,0,0.3)',
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border-light)',
                    borderRadius: 6,
                    padding: 8,
                    fontFamily: 'monospace',
                    fontSize: 11,
                    resize: 'vertical',
                  }}
                  onFocus={(e) => e.target.select()}
                />
              </details>
            </div>
          )}

          <div className="spotify-rail-grid">
            {spotifyTracks.map((t) => (
              <div key={t.spotify_id} className="spotify-card">
                <div className="spotify-card-art">
                  {t.album_art_url ? (
                    <img src={t.album_art_url} alt={t.album} loading="lazy" />
                  ) : (
                    <div className="spotify-card-art-fallback">♪</div>
                  )}
                </div>
                <div className="spotify-card-body">
                  <div className="spotify-card-title" title={t.name}>{t.name}</div>
                  <div className="spotify-card-artists" title={t.artists.join(', ')}>
                    {t.artists.join(', ') || 'Unknown artist'}
                  </div>
                  <a
                    className="spotify-card-link"
                    href={t.external_url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Open in Spotify ↗
                  </a>
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* ── LOCAL MODE RESULTS ── */}
      {source === 'local' && playlist.length > 0 && (
        <>
          <div className="section-header">
            <div className="section-title">
              {playlist.length} tracks · {mode === 'emotion' ? targetEmotion : `your ${profile?.dominant_emotion} profile`}
            </div>
            <button className="btn btn-ghost" onClick={exportCSV} style={{ fontSize: 12 }}>
              ⬇ Export CSV
            </button>
          </div>

          <div className="track-list">
            {playlist.map((track) => (
              <div key={track.path} className="track-row">
                <span className="track-number">{track.rank}</span>
                <div className="track-info">
                  <div className="track-name">{track.track_name}</div>
                  <div className="track-meta">
                    {Object.entries(track.emotion_scores).map(([e, s]) => (
                      <span key={e}>{e} {(s * 100).toFixed(0)}%</span>
                    ))}
                  </div>
                </div>
                <span className={`track-emotion-badge badge-${track.dominant_emotion}`}>
                  {EMOTION_EMOJIS[track.dominant_emotion]} {track.dominant_emotion}
                </span>
                <span className="track-score">{track.relevance_score.toFixed(2)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
