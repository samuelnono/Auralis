import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import Analyze from './pages/Analyze'
import Recommendations from './pages/Recommendations'
import Playlist from './pages/Playlist'
import Chat from './pages/Chat'
import Profile from './pages/Profile'
import SpotifyCallback from './pages/SpotifyCallback'
import { API } from './config'
import './App.css'

/**
 * Local-storage-backed React state. Used for the sidebar collapse preference
 * so the app remembers whether the user prefers the rail open or closed
 * across reloads.
 */
function useStickyState(key, initial) {
  const [value, setValue] = useState(() => {
    try {
      const raw = localStorage.getItem(key)
      return raw !== null ? JSON.parse(raw) : initial
    } catch {
      return initial
    }
  })
  useEffect(() => {
    try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* quota — ignore */ }
  }, [key, value])
  return [value, setValue]
}

function App() {
  const [profile, setProfile] = useState(null)
  const [collapsed, setCollapsed] = useStickyState('auralis.sidebar.collapsed', false)

  const fetchProfile = async () => {
    try {
      const res = await fetch(`${API}/profile`)
      const data = await res.json()
      setProfile(data)
    } catch (e) {}
  }

  useEffect(() => { fetchProfile() }, [])

  return (
    <Router>
      <div className={`app ${collapsed ? 'sidebar-collapsed' : ''}`}>
        <nav className="sidebar">
          <button
            className="sidebar-toggle"
            onClick={() => setCollapsed((c) => !c)}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {/* Hamburger / chevron — same icon flips meaning based on state */}
            <span className="sidebar-toggle-icon" aria-hidden="true">
              {collapsed ? '☰' : '⟨'}
            </span>
          </button>

          <div className="sidebar-logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">Auralis</span>
          </div>
          <div className="sidebar-tagline">Emotion-Aware Music</div>

          <div className="nav-links">
            <NavLink to="/" end className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'} title="Analyze">
              <span className="nav-icon">⟁</span> <span className="nav-label">Analyze</span>
            </NavLink>
            <NavLink to="/recommendations" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'} title="For You">
              <span className="nav-icon">✦</span> <span className="nav-label">For You</span>
            </NavLink>
            <NavLink to="/playlist" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'} title="Playlist">
              <span className="nav-icon">⋮⋮</span> <span className="nav-label">Playlist</span>
            </NavLink>
            <NavLink to="/chat" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'} title="Chat">
              <span className="nav-icon">◎</span> <span className="nav-label">Chat</span>
            </NavLink>
            <NavLink to="/profile" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'} title="Profile">
              <span className="nav-icon">◉</span> <span className="nav-label">Profile</span>
            </NavLink>
          </div>

          {profile?.has_signal && (
            <div className="sidebar-profile">
              <div className="profile-label">Your vibe</div>
              <div className="profile-emotion">{profile.dominant_emotion}</div>
              <div className="profile-stats">
                <span>♥ {profile.total_likes}</span>
                <span>✕ {profile.total_dislikes}</span>
              </div>
              <div className="affinity-bars">
                {Object.entries(profile.emotion_affinity || {}).map(([emotion, score]) => (
                  <div key={emotion} className="affinity-row">
                    <span className="affinity-label">{emotion}</span>
                    <div className="affinity-bar-track">
                      <div className="affinity-bar-fill" style={{width: `${score * 100}%`}} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </nav>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<Analyze onFeedback={fetchProfile} />} />
            <Route path="/recommendations" element={<Recommendations profile={profile} onFeedback={fetchProfile} />} />
            <Route path="/playlist" element={<Playlist profile={profile} onFeedback={fetchProfile} />} />
            <Route path="/chat" element={<Chat onFeedback={fetchProfile} />} />
            <Route path="/profile" element={<Profile profile={profile} onReset={fetchProfile} onFeedback={fetchProfile} />} />
            <Route path="/callback" element={<SpotifyCallback />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
