/**
 * SpotifyRail — drops a strip of live Spotify tracks into any page that
 * already knows the listener's mood point. Surfaces results from the
 * backend's /spotify/search-by-mood endpoint, which translates a
 * (valence, arousal) pair (or a discrete emotion label) into a Search
 * query under the hood.
 *
 * Phase 1: read-only discovery. The "Open in Spotify" external link works
 *          for everyone, even non-Premium listeners.
 * Phase 2: a "+ Save to Library" button will appear once the user has
 *          authenticated via PKCE in the Profile page.
 */

import { useEffect, useState, useRef } from 'react'
import axios from 'axios'
import { API } from '../config'
import { useSpotifyAuth } from '../hooks/useSpotifyAuth'
import { saveTracks } from '../lib/spotifyApi'

// Spotify's Feb 2026 Dev-Mode tightening blocks PUT /me/tracks (Liked
// Songs writes) until the app is approved for Extended Quota. While we
// wait for that approval, the heart on a Spotify card writes a like
// only to the Auralis profile (which always works) and we suppress the
// Spotify-side library call to avoid a noisy 403 in the network tab.
// Flip this back to true once Spotify approves Auralis for production.
const WRITE_TO_SPOTIFY_LIBRARY = false

function PreviewButton({ previewUrl }) {
  const audioRef = useRef(null)
  const [playing, setPlaying] = useState(false)

  // Spotify is winding down preview_url for many tracks, so we render a
  // disabled placeholder rather than hiding the button entirely. Keeps
  // the card layout stable across rows.
  if (!previewUrl) {
    return (
      <button
        className="spotify-preview-btn spotify-preview-btn-disabled"
        title="No preview available for this track"
        disabled
      >
        ▶
      </button>
    )
  }

  const toggle = () => {
    if (!audioRef.current) {
      audioRef.current = new Audio(previewUrl)
      audioRef.current.addEventListener('ended', () => setPlaying(false))
    }
    if (playing) {
      audioRef.current.pause()
      setPlaying(false)
    } else {
      audioRef.current.play()
      setPlaying(true)
    }
  }

  // Always pause on unmount so a user navigating away doesn't keep audio going.
  useEffect(() => () => audioRef.current?.pause(), [])

  return (
    <button
      className={`spotify-preview-btn ${playing ? 'playing' : ''}`}
      onClick={toggle}
      title={playing ? 'Pause preview' : 'Play 30s preview'}
    >
      {playing ? '❚❚' : '▶'}
    </button>
  )
}


/**
 * Heart-style "+ Save" button. Renders only for logged-in users.
 *
 * On click it always records a like against the user's Auralis profile via
 * POST /spotify/feedback — that flow is fully under our control and is what
 * makes Spotify-card likes show up in Profile → Interaction History.
 *
 * It additionally writes to the user's Spotify Liked Songs (PUT /me/tracks)
 * IFF ``WRITE_TO_SPOTIFY_LIBRARY`` is true. That flag is currently off
 * because Spotify Dev-Mode blocks the call; flipping it on after production
 * approval restores the dual-write behaviour with no other code changes.
 *
 * The heart fills optimistically on click and, since the Spotify-side path
 * is gated, behaves as a one-shot like for now (no un-heart). Once the
 * Spotify path is enabled again we'll restore the toggle semantics.
 */
function SaveButton({ track, isLoggedIn, initiallySaved, savedReady = true, discreteEmotion = null, onFeedback = null }) {
  const [saved, setSaved] = useState(!!initiallySaved)
  const [busy, setBusy]   = useState(false)
  const [error, setError] = useState(null)

  // Keep the button in sync if the parent's saved-state lookup arrives later.
  useEffect(() => { setSaved(!!initiallySaved) }, [initiallySaved])

  if (!isLoggedIn || !track?.spotify_id) return null

  const onClick = async () => {
    // Disable clicks until the parent's "is this already liked?" lookup
    // has resolved — otherwise an early click on a will-be-pre-filled
    // heart records a duplicate like (which is what produced the
    // "Marvellous" appearing twice in interaction history).
    if (busy || saved || !savedReady) return
    setBusy(true)
    setError(null)
    setSaved(true) // optimistic — the Auralis call is reliable, the Spotify call is gated

    // Auralis-profile write (always). Failure here rolls back the heart.
    try {
      await axios.post(`${API}/spotify/feedback`, {
        spotify_id:       track.spotify_id,
        track_name:       track.name,
        artists:          track.artists || [],
        label:            'like',
        discrete_emotion: discreteEmotion,
      })
    } catch (e) {
      setSaved(false)
      setError(e.response?.data?.detail || e.message || 'Could not record like.')
      setBusy(false)
      return
    }

    // Notify the app shell so the global profile state (Vibe sidebar +
    // Profile page Interaction History) refreshes right after the like
    // lands. Without this, the user has to reset/reload to see their new
    // Spotify hearts reflected on the Profile tab.
    if (typeof onFeedback === 'function') {
      try { onFeedback() } catch { /* swallow — the like already saved */ }
    }

    // Spotify-library write (gated). Failure here is silent — the Auralis
    // record already succeeded so the heart stays filled.
    if (WRITE_TO_SPOTIFY_LIBRARY) {
      try {
        await saveTracks([track.spotify_id])
      } catch {
        /* ignore — Auralis-side like already counted */
      }
    }

    setBusy(false)
  }

  return (
    <button
      className={`spotify-save-btn ${saved ? 'saved' : ''}`}
      onClick={onClick}
      disabled={busy || saved || !savedReady}
      title={
        error ? error
        : !savedReady ? 'Loading…'
        : saved ? 'Liked — recorded in your Auralis profile'
        : 'Like this track (records to your Auralis taste profile)'
      }
      aria-label={saved ? 'Liked' : 'Like this track'}
    >
      {saved ? '♥' : '♡'}
    </button>
  )
}


function SpotifyTrackCard({ track, isLoggedIn, initiallySaved, savedReady = true, discreteEmotion = null, onFeedback = null }) {
  return (
    <div className="spotify-card">
      <div className="spotify-card-art">
        {track.album_art_url ? (
          <img src={track.album_art_url} alt={track.album} loading="lazy" />
        ) : (
          <div className="spotify-card-art-fallback">♪</div>
        )}
        <PreviewButton previewUrl={track.preview_url} />
      </div>
      <div className="spotify-card-body">
        <div className="spotify-card-title" title={track.name}>{track.name}</div>
        <div className="spotify-card-artists" title={track.artists.join(', ')}>
          {track.artists.join(', ') || 'Unknown artist'}
        </div>
        <div className="spotify-card-actions">
          <a
            className="spotify-card-link"
            href={track.external_url}
            target="_blank"
            rel="noopener noreferrer"
          >
            Open in Spotify ↗
          </a>
          <SaveButton
            track={track}
            isLoggedIn={isLoggedIn}
            initiallySaved={initiallySaved}
            savedReady={savedReady}
            discreteEmotion={discreteEmotion}
            onFeedback={onFeedback}
          />
        </div>
      </div>
    </div>
  )
}


export default function SpotifyRail({
  valence = null,
  arousal = null,
  discreteEmotion = null,
  limit = 6,
  title = 'Real tracks for this mood',
  subtitle = null,
  onFeedback = null,
  seedArtists = [],
  seedGenres = [],
}) {
  const [tracks, setTracks] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [meta, setMeta] = useState(null)
  // savedMap[spotify_id] = boolean. Populated from /profile/spotify-likes
  // once the rail's tracks have rendered. ``savedReady`` flips to true
  // once that lookup resolves so heart clicks are gated until then —
  // prevents the race condition where an early click on a track that
  // *will* be pre-filled records a duplicate like.
  const [savedMap, setSavedMap] = useState({})
  const [savedReady, setSavedReady] = useState(false)

  const { isLoggedIn } = useSpotifyAuth()

  // Re-fetch whenever the mood point shifts. We round to 2 decimals so
  // tiny numerical jitter from re-analysing the same file doesn't trigger
  // an unnecessary network call.
  const moodKey = [
    valence !== null ? Number(valence).toFixed(2) : 'x',
    arousal !== null ? Number(arousal).toFixed(2) : 'x',
    discreteEmotion || 'x',
  ].join('|')

  useEffect(() => {
    const hasMood = (valence !== null && arousal !== null) || !!discreteEmotion
    if (!hasMood) return

    let cancelled = false
    setLoading(true)
    setError(null)

    axios
      .post(`${API}/spotify/search-by-mood`, {
        valence,
        arousal,
        discrete_emotion: discreteEmotion,
        limit,
        seed_artists: seedArtists,
        seed_genres:  seedGenres,
      })
      .then((res) => {
        if (cancelled) return
        const fetched = res.data.tracks || []
        setTracks(fetched)
        setMeta({
          query: res.data.query,
          quadrant: res.data.quadrant,
          rationale: res.data.rationale,
        })
        // Resolve heart-fill state from the Auralis profile (which tracks
        // the user has liked) rather than Spotify's library — Spotify's
        // `/me/tracks/contains` requires `user-library-read` (not in our
        // scope set) AND its writes are blocked in dev mode anyway, so
        // querying the Auralis side both fixes the persistence bug and
        // sidesteps the silent 403s. Failures here are non-fatal.
        setSavedReady(false)
        axios
          .get(`${API}/profile/spotify-likes`)
          .then((res) => {
            if (cancelled) return
            const liked = new Set(res.data?.liked || [])
            const next = {}
            for (const id of fetched.map((t) => t.spotify_id).filter(Boolean)) {
              next[id] = liked.has(id)
            }
            setSavedMap(next)
            setSavedReady(true)
          })
          .catch(() => {
            // Default to "ready with empty likes" rather than locking the
            // hearts forever if the lookup fails.
            if (!cancelled) setSavedReady(true)
          })
      })
      .catch((e) => {
        if (cancelled) return
        const detail = e.response?.data?.detail
        if (e.response?.status === 503) {
          setError('Spotify temporarily unavailable. Try again in a moment.')
        } else if (typeof detail === 'string' && detail.includes('credentials')) {
          setError('Spotify is not configured on this server.')
        } else {
          setError('Could not load Spotify tracks.')
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })

    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
    // Re-running when the seeds change is intentional: the rail loads
    // initially with no seeds (auth still resolving), then re-fetches
    // once the user's top-artist/genre seeds arrive so the picks land
    // in their musical neighborhood. Stringifying keeps React from
    // misfiring on every render due to array identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [moodKey, limit, isLoggedIn, seedArtists.join('|'), seedGenres.join('|')])

  const hasMood = (valence !== null && arousal !== null) || !!discreteEmotion
  if (!hasMood) return null

  return (
    <div className="spotify-rail">
      <div className="spotify-rail-header">
        <div>
          <div className="spotify-rail-title">{title}</div>
          {(subtitle || meta?.rationale) && (
            <div className="spotify-rail-subtitle">
              {subtitle || meta?.rationale}
            </div>
          )}
        </div>
        <div className="spotify-rail-badge" aria-hidden="true">
          Spotify
        </div>
      </div>

      {loading && (
        <div className="loading" style={{ padding: '24px 0' }}>
          <div className="spinner" /> Finding Spotify matches...
        </div>
      )}

      {error && !loading && (
        <div className="spotify-rail-error">{error}</div>
      )}

      {!loading && !error && tracks.length === 0 && (
        <div className="spotify-rail-empty">No tracks came back for this mood.</div>
      )}

      {!loading && !error && tracks.length > 0 && (
        <div className="spotify-rail-grid">
          {tracks.map((t) => (
            <SpotifyTrackCard
              key={t.spotify_id}
              track={t}
              isLoggedIn={isLoggedIn}
              initiallySaved={savedMap[t.spotify_id]}
              savedReady={savedReady}
              discreteEmotion={discreteEmotion}
              onFeedback={onFeedback}
            />
          ))}
        </div>
      )}
    </div>
  )
}
