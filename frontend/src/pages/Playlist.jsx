import { useState } from 'react'
import axios from 'axios'
import { API } from '../config'

const EMOTION_EMOJIS = { calm: '🌊', energetic: '⚡', happy: '☀️', sad: '🌧️' }
const MOOD_DESCRIPTIONS = {
  calm: 'Low tempo, smooth textures, gentle energy',
  energetic: 'High BPM, dynamic range, driving rhythms',
  happy: 'Bright timbre, uplifting spectral centroid',
  sad: 'Low brightness, minor spectral patterns',
}

export default function Playlist({ profile }) {
  const [mode, setMode] = useState('emotion')
  const [targetEmotion, setTargetEmotion] = useState('calm')
  const [length, setLength] = useState(8)
  const [playlist, setPlaylist] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const generate = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await axios.post(`${API}/playlist`, {
        mode,
        target_emotion: targetEmotion,
        length,
      })
      setPlaylist(res.data.playlist)
    } catch (e) {
      setError(e.response?.data?.detail || 'Failed to generate playlist')
    } finally {
      setLoading(false)
    }
  }

  const exportCSV = () => {
    window.open(`${API}/playlist/export?mode=${mode}&target_emotion=${targetEmotion}&length=${length}`)
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Playlist</h1>
        <p className="page-subtitle">Generate an emotion-aware playlist from your indexed collection.</p>
      </div>

      {/* Mode selector */}
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
          <input type="range" min={3} max={20} step={1} value={length}
            onChange={e => setLength(parseInt(e.target.value))} />
        </div>
      </div>

      <button className="btn btn-primary" onClick={generate} disabled={loading}
        style={{ marginBottom: 32, width: '100%' }}>
        {loading ? '⟳ Generating...' : '⋮⋮ Generate Playlist'}
      </button>

      {loading && <div className="loading"><div className="spinner" /> Building playlist...</div>}
      {error && <div style={{ color: '#ff6b8a', fontSize: 13, marginBottom: 16 }}>{error}</div>}

      {playlist.length > 0 && (
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
