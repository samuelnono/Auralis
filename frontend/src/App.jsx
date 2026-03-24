import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import Analyze from './pages/Analyze'
import Recommendations from './pages/Recommendations'
import Playlist from './pages/Playlist'
import Chat from './pages/Chat'
import Profile from './pages/Profile'
import './App.css'

function App() {
  const [profile, setProfile] = useState(null)

  const fetchProfile = async () => {
    try {
      const res = await fetch('http://localhost:8000/profile')
      const data = await res.json()
      setProfile(data)
    } catch (e) {}
  }

  useEffect(() => { fetchProfile() }, [])

  return (
    <Router>
      <div className="app">
        <nav className="sidebar">
          <div className="sidebar-logo">
            <span className="logo-icon">◈</span>
            <span className="logo-text">Auralis</span>
          </div>
          <div className="sidebar-tagline">Emotion-Aware Music</div>

          <div className="nav-links">
            <NavLink to="/" end className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <span className="nav-icon">⟁</span> Analyze
            </NavLink>
            <NavLink to="/recommendations" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <span className="nav-icon">✦</span> For You
            </NavLink>
            <NavLink to="/playlist" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <span className="nav-icon">⋮⋮</span> Playlist
            </NavLink>
            <NavLink to="/chat" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <span className="nav-icon">◎</span> Chat
            </NavLink>
            <NavLink to="/profile" className={({isActive}) => isActive ? 'nav-item active' : 'nav-item'}>
              <span className="nav-icon">◉</span> Profile
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
            <Route path="/recommendations" element={<Recommendations profile={profile} />} />
            <Route path="/playlist" element={<Playlist profile={profile} />} />
            <Route path="/chat" element={<Chat />} />
            <Route path="/profile" element={<Profile profile={profile} onReset={fetchProfile} />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
