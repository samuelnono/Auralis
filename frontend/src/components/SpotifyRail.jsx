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


function SpotifyTrackCard({ track }) {
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
        <a
          className="spotify-card-link"
          href={track.external_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          Open in Spotify ↗
        </a>
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
}) {
  const [tracks, setTracks] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [meta, setMeta] = useState(null)

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
      })
      .then((res) => {
        if (cancelled) return
        setTracks(res.data.tracks || [])
        setMeta({
          query: res.data.query,
          quadrant: res.data.quadrant,
          rationale: res.data.rationale,
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
  }, [moodKey, limit])

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
            <SpotifyTrackCard key={t.spotify_id} track={t} />
          ))}
        </div>
      )}
    </div>
  )
}
