import { useState } from 'react'
import axios from 'axios'
import { API } from '../config'

const EMOTION_COLORS = { calm: 'var(--calm)', energetic: 'var(--energetic)', happy: 'var(--happy)', sad: 'var(--sad)' }

export default function Profile({ profile, onReset }) {
  const [resetting, setResetting] = useState(false)

  const handleReset = async () => {
    if (!confirm('Reset your profile? This cannot be undone.')) return
    setResetting(true)
    await axios.delete(`${API}/profile`)
    onReset()
    setResetting(false)
  }

  if (!profile?.has_signal) {
    return (
      <div className="page">
        <div className="page-header">
          <h1 className="page-title">Profile</h1>
        </div>
        <div className="empty-state">
          <div className="empty-state-icon">◉</div>
          <div className="empty-state-title">No profile yet</div>
          <div className="empty-state-text">Rate tracks in Analyze to build your taste profile.</div>
        </div>
      </div>
    )
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Your Profile</h1>
        <p className="page-subtitle">Your accumulated music taste — built from everything you've rated.</p>
      </div>

      {/* Stats */}
      <div className="stat-row" style={{ marginBottom: 32 }}>
        <div className="stat-pill">
          <span className="stat-pill-label">Liked</span>
          <span className="stat-pill-value" style={{ color: '#ff6b8a' }}>{profile.total_likes}</span>
        </div>
        <div className="stat-pill">
          <span className="stat-pill-label">Disliked</span>
          <span className="stat-pill-value">{profile.total_dislikes}</span>
        </div>
        <div className="stat-pill">
          <span className="stat-pill-label">Total Ratings</span>
          <span className="stat-pill-value">{profile.total_likes + profile.total_dislikes}</span>
        </div>
        <div className="stat-pill">
          <span className="stat-pill-label">Dominant Vibe</span>
          <span className="stat-pill-value" style={{ color: 'var(--accent)', textTransform: 'capitalize' }}>
            {profile.dominant_emotion}
          </span>
        </div>
      </div>

      {/* Emotion Affinity */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="section-title" style={{ marginBottom: 20 }}>Emotion Affinity</div>
        <div style={{ display: 'flex', flex: 1, gap: 12, alignItems: 'flex-end', height: 120 }}>
          {Object.entries(profile.emotion_affinity || {}).map(([emotion, score]) => (
            <div key={emotion} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{(score * 100).toFixed(0)}%</div>
              <div style={{
                width: '100%',
                height: `${Math.max(score * 100, 4)}px`,
                background: EMOTION_COLORS[emotion] || 'var(--border-light)',
                borderRadius: '4px 4px 0 0',
                transition: 'height 0.5s ease',
                opacity: 0.8,
              }} />
              <div style={{ fontSize: 11, color: 'var(--text-secondary)', textTransform: 'capitalize' }}>{emotion}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Interaction History */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="section-title" style={{ marginBottom: 16 }}>Interaction History</div>
        <div className="track-list">
          {(profile.interaction_log || []).slice().reverse().map((entry, i) => (
            <div key={i} className="track-row" style={{ cursor: 'default' }}>
              <span style={{ fontSize: 14 }}>{entry.feedback === 'like' ? '♥' : '✕'}</span>
              <div className="track-info">
                <div className="track-name">
                  {entry.track.split(/[\\/]/).pop().replace(/\.[^.]+$/, '')}
                </div>
              </div>
              <span className={`track-emotion-badge badge-${entry.emotion_label}`}>
                {entry.emotion_label}
              </span>
              <span style={{ fontSize: 11, color: entry.feedback === 'like' ? '#ff6b8a' : 'var(--text-muted)' }}>
                {entry.feedback}
              </span>
            </div>
          ))}
        </div>
      </div>

      <button className="btn btn-ghost" onClick={handleReset} disabled={resetting}
        style={{ color: '#ff6b8a', borderColor: 'rgba(255,107,138,0.3)' }}>
        {resetting ? 'Resetting...' : '🗑 Reset Profile'}
      </button>
    </div>
  )
}
