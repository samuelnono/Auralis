import { useState, useEffect } from 'react'
import axios from 'axios'
import { API } from '../config'

const EMOTION_EMOJIS = { calm: '🌊', energetic: '⚡', happy: '☀️', sad: '🌧️', unknown: '🎵' }

export default function Recommendations({ profile }) {
  const [recs, setRecs] = useState([])
  const [loading, setLoading] = useState(false)
  const [alpha, setAlpha] = useState(0.7)
  const [topK, setTopK] = useState(10)
  const [excludeRated, setExcludeRated] = useState(true)
  const [error, setError] = useState(null)

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

  if (!profile?.has_signal) {
    return (
      <div className="page">
        <div className="page-header">
          <h1 className="page-title">For You</h1>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">✦</div>
          <div className="empty-state-title">No profile yet</div>
          <div className="empty-state-text">Go to Analyze, upload a track, and rate it to start getting recommendations.</div>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">For You</h1>
        <p className="page-subtitle">
          Ranked by your preference profile — dominant taste: <span style={{ color: 'var(--accent)' }}>{profile.dominant_emotion}</span>
        </p>
      </div>

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
  )
}
