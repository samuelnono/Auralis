import { useState, useEffect } from 'react'
import axios from 'axios'
import { API } from '../config'
import SpotifyRail from '../components/SpotifyRail'
import SpotifyTasteRail from '../components/SpotifyTasteRail'
import { useSpotifyAuth } from '../hooks/useSpotifyAuth'
import { getTopArtists } from '../lib/spotifyApi'

const EMOTION_EMOJIS = { calm: '🌊', energetic: '⚡', happy: '☀️', sad: '🌧️', unknown: '🎵' }

export default function Recommendations({ profile, onFeedback = null }) {
  const [recs, setRecs] = useState([])
  const [loading, setLoading] = useState(false)
  const [alpha, setAlpha] = useState(0.7)
  const [topK, setTopK] = useState(10)
  const [excludeRated, setExcludeRated] = useState(true)
  const [error, setError] = useState(null)
  // Collapsed by default — these are research-index matches against the
  // training dataset, useful for explaining the model but noisy as a primary
  // recommendation surface for end users.
  const [showResearchIndex, setShowResearchIndex] = useState(false)

  // Spotify seeds shared by both rails. Lifted up here (rather than fetched
  // independently in each rail) so the mood-search rail uses the same
  // artist + genre context as the chat — keeps recommendations coherent
  // across surfaces instead of one rail being mood-only and another being
  // mood + taste-blended.
  const { isLoggedIn: spotifyConnectedAuth } = useSpotifyAuth()
  const [seedArtists, setSeedArtists] = useState([])
  const [seedGenres,  setSeedGenres]  = useState([])
  useEffect(() => {
    let cancelled = false
    if (!spotifyConnectedAuth) {
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
        const seen = new Set()
        const genres = []
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
      } catch {
        if (!cancelled) {
          setSeedArtists([])
          setSeedGenres([])
        }
      }
    })()
    return () => { cancelled = true }
  }, [spotifyConnectedAuth])

  const fetchRecs = async () => {
    if (!profile?.has_signal) return
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API}/recommendations`, {
        alpha, top_k: topK, exclude_rated: excludeRated
      })
      setRecs(res.data.recommendations)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to load recommendations')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchRecs() }, [profile, alpha, topK, excludeRated])

  const { isLoggedIn: spotifyConnected } = useSpotifyAuth()
  const hasLocalProfile = !!profile?.has_signal

  // Empty state only fires when *both* signals are missing — without a local
  // profile *and* without a Spotify connection we have nothing to recommend
  // off, so steer the user toward providing one or the other.
  if (!hasLocalProfile && !spotifyConnected) {
    return (
      <div className="page">
        <div className="page-header">
          <h1 className="page-title">For You</h1>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">✦</div>
          <div className="empty-state-title">No profile yet</div>
          <div className="empty-state-text">
            Connect Spotify on the Profile page to recommend from your listening history,
            or analyze and rate a track to build a local taste profile.
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">For You</h1>
        <p className="page-subtitle">
          {hasLocalProfile && spotifyConnected
            ? <>Mixing your Spotify history with your locally-rated <span style={{ color: 'var(--accent)' }}>{profile.dominant_emotion}</span> taste.</>
            : hasLocalProfile
              ? <>Ranked by your preference profile — dominant taste: <span style={{ color: 'var(--accent)' }}>{profile.dominant_emotion}</span></>
              : <>Recommendations driven by your Spotify listening history.</>}
        </p>
      </div>

      {/* Spotify-driven rail — built from the user's actual top artists,
          optionally filtered by the local emotion profile when both signals
          are present. Renders nothing when the user isn't connected. */}
      <SpotifyTasteRail
        emotion={hasLocalProfile ? profile.dominant_emotion : null}
        title={hasLocalProfile
          ? 'Your most-played on Spotify'
          : 'Based on your Spotify taste'}
        limit={8}
        onFeedback={onFeedback}
      />

      {/* Mood-based rail — driven by the local rating profile, but blended
          with the user's Spotify top artists + genres so the picks land in
          the same musical neighborhood the chat recommends from. Skipped
          for users who only have a Spotify connection, since the rail
          above already covers them. */}
      {hasLocalProfile && (
        <SpotifyRail
          discreteEmotion={profile.dominant_emotion}
          title={`Spotify picks for your ${profile.dominant_emotion} mood`}
          limit={6}
          onFeedback={onFeedback}
          seedArtists={seedArtists}
          seedGenres={seedGenres}
        />
      )}

      {/* Research-index ranking, collapsed by default. This is the
          model's similarity ranking against the training dataset
          (Song_01..Song_N MFCC fingerprints) — useful for showing how the
          AI pipeline works, less useful as a primary recommendation surface
          since the dataset tracks aren't user-playable.

          Hidden entirely for users who only have a Spotify connection — the
          research index requires a local rating profile to score against. */}
      {hasLocalProfile && (
      <div className="research-section">
        <button
          className="research-toggle"
          onClick={() => setShowResearchIndex((s) => !s)}
        >
          <span className="research-toggle-icon">
            {showResearchIndex ? '▾' : '▸'}
          </span>
          <span className="research-toggle-label">
            {showResearchIndex ? 'Hide' : 'Show'} research-index matches
          </span>
          <span className="research-toggle-hint">
            (model similarity against the training dataset)
          </span>
        </button>

        {showResearchIndex && (
          <div className="research-body">
            {/* Controls */}
            <div className="card" style={{ marginBottom: 28 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
                <div className="slider-row">
                  <div className="slider-label">
                    <span>Acoustic ↔ Emotion weight</span>
                    <span className="slider-value">{alpha.toFixed(2)}</span>
                  </div>
                  <input type="range" min={0} max={1} step={0.05} value={alpha}
                    onChange={e => setAlpha(parseFloat(e.target.value))} />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-muted)' }}>
                    <span>Emotion only</span><span>Acoustic only</span>
                  </div>
                </div>

                <div className="slider-row">
                  <div className="slider-label">
                    <span>Results</span>
                    <span className="slider-value">{topK}</span>
                  </div>
                  <input type="range" min={3} max={20} step={1} value={topK}
                    onChange={e => setTopK(parseInt(e.target.value))} />
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginTop: 12 }}>
                <input type="checkbox" id="excludeRated" checked={excludeRated}
                  onChange={e => setExcludeRated(e.target.checked)}
                  style={{ accentColor: 'var(--accent)', cursor: 'pointer' }} />
                <label htmlFor="excludeRated" style={{ fontSize: 12, color: 'var(--text-secondary)', cursor: 'pointer' }}>
                  Hide already-rated tracks
                </label>
              </div>
            </div>

            {loading && <div className="loading"><div className="spinner" /> Finding matches...</div>}
            {error && <div style={{ color: '#ff6b8a', fontSize: 13, marginBottom: 16 }}>{error}</div>}

            {recs.length === 0 && !loading && (
              <div className="empty-state">
                <div className="empty-state-icon">◎</div>
                <div className="empty-state-title">No results</div>
                <div className="empty-state-text">Try unchecking "Hide already-rated tracks" or rate more songs.</div>
              </div>
            )}

            <div className="track-list">
              {recs.map((rec, i) => (
                <div key={rec.path} className="track-row">
                  <span className="track-number">{i + 1}</span>
                  <div className="track-info">
                    <div className="track-name">{rec.path.split(/[\\/]/).pop().replace(/\.[^.]+$/, '')}</div>
                    <div className="track-meta">
                      <span>Acoustic {(rec.mfcc_sim * 100).toFixed(0)}%</span>
                      <span>·</span>
                      <span>Emotion {(rec.emotion_sim * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                  <span className={`track-emotion-badge badge-${rec.emotion}`}>
                    {EMOTION_EMOJIS[rec.emotion]} {rec.emotion}
                  </span>
                  <span className="track-score">{rec.blended_score.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
      )}
    </div>
  )
}
