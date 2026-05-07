import { useState } from 'react'
import axios from 'axios'
import { API } from '../config'
import { useSpotifyAuth } from '../hooks/useSpotifyAuth'

const EMOTION_COLORS = { calm: 'var(--calm)', energetic: 'var(--energetic)', happy: 'var(--happy)', sad: 'var(--sad)' }

/**
 * The Profile page now has two distinct sections:
 *   1. Spotify connection card — always visible. Lets the user log in / out
 *      regardless of whether they've rated any tracks yet, because Save and
 *      "Push to Spotify" need the connection independent of the local taste
 *      profile.
 *   2. Taste profile — only shown after the user has rated something locally.
 */
export default function Profile({ profile, onReset }) {
  const [resetting, setResetting] = useState(false)
  const spotify = useSpotifyAuth()

  const handleReset = async () => {
    if (!confirm('Reset your profile? This cannot be undone.')) return
    setResetting(true)
    await axios.delete(`${API}/profile`)
    onReset()
    setResetting(false)
  }

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Your Profile</h1>
        <p className="page-subtitle">
          Connect Spotify to save tracks and push playlists, then rate songs in Analyze
          to build your taste profile.
        </p>
      </div>

      {/* Spotify connection card — always visible. */}
      <SpotifyConnectionCard spotify={spotify} />

      {/* Taste profile — only meaningful once the user has rated something. */}
      {profile?.has_signal ? (
        <>
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
              {(profile.interaction_log || []).slice().reverse().map((entry, i) => {
                // Prefer the original upload filename. For old log entries
                // (recorded before display_name was plumbed through) fall back
                // to stripping the tmp directory off the path. As a last
                // resort show a generic "Track N" so the row never reads as
                // an opaque tmp hash.
                const stem = (entry.track || '').split(/[\\/]/).pop().replace(/\.[^.]+$/, '')
                const looksLikeTmp = /^tmp[a-z0-9]+$/i.test(stem)
                const label = entry.display_name?.replace(/\.[^.]+$/, '')
                  || (looksLikeTmp ? `Track ${profile.interaction_log.length - i}` : stem)
                return (
                <div key={i} className="track-row" style={{ cursor: 'default' }}>
                  <span style={{ fontSize: 14 }}>{entry.feedback === 'like' ? '♥' : '✕'}</span>
                  <div className="track-info">
                    <div className="track-name">{label}</div>
                  </div>
                  <span className={`track-emotion-badge badge-${entry.emotion_label}`}>
                    {entry.emotion_label}
                  </span>
                  <span style={{ fontSize: 11, color: entry.feedback === 'like' ? '#ff6b8a' : 'var(--text-muted)' }}>
                    {entry.feedback}
                  </span>
                </div>
                )
              })}
            </div>
          </div>

          <button className="btn btn-ghost" onClick={handleReset} disabled={resetting}
            style={{ color: '#ff6b8a', borderColor: 'rgba(255,107,138,0.3)' }}>
            {resetting ? 'Resetting...' : '🗑 Reset Profile'}
          </button>
        </>
      ) : (
        <div className="empty-state">
          <div className="empty-state-icon">◉</div>
          <div className="empty-state-title">No taste profile yet</div>
          <div className="empty-state-text">Rate tracks in Analyze to build your taste profile.</div>
        </div>
      )}
    </div>
  )
}


/**
 * Card-style block that shows either the "Connect Spotify" CTA or the
 * connected user's identity + a Disconnect button. Kept inline in this file
 * because it's only used here.
 */
function SpotifyConnectionCard({ spotify }) {
  const { user, isLoggedIn, loading, login, logout } = spotify

  if (loading) {
    return (
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="loading"><div className="spinner" /> Checking Spotify connection…</div>
      </div>
    )
  }

  if (!isLoggedIn) {
    return (
      <div className="card" style={{ marginBottom: 28 }}>
        <div className="section-title" style={{ marginBottom: 8 }}>Spotify connection</div>
        <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 16, lineHeight: 1.5 }}>
          Connect your Spotify account to save Auralis recommendations to your library
          and push generated playlists straight to your account.
        </p>
        <button className="btn btn-primary" onClick={login} style={{ background: '#1ed760', borderColor: '#1ed760', color: '#000' }}>
          Connect Spotify
        </button>
        <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 12 }}>
          Auralis is in Spotify Developer Mode while we wait for production approval —
          only emails added to the dashboard's user list can sign in.
        </p>
      </div>
    )
  }

  const avatar = user?.images?.[0]?.url
  const display = user?.display_name || user?.id || 'Spotify user'
  const email = user?.email

  return (
    <div className="card" style={{ marginBottom: 28 }}>
      <div className="section-title" style={{ marginBottom: 16 }}>Spotify connection</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
        {avatar ? (
          <img src={avatar} alt="" style={{ width: 56, height: 56, borderRadius: '50%', objectFit: 'cover' }} />
        ) : (
          <div style={{
            width: 56, height: 56, borderRadius: '50%',
            background: '#1ed760', color: '#000',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 22, fontWeight: 700,
          }}>
            {display.charAt(0).toUpperCase()}
          </div>
        )}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 15, fontWeight: 600 }}>Connected as {display}</div>
          {email && <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{email}</div>}
        </div>
        <button className="btn btn-ghost" onClick={logout}>Disconnect</button>
      </div>
    </div>
  )
}
