import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { API } from '../config'
import { useSpotifyAuth } from '../hooks/useSpotifyAuth'
import { getTopArtists, getTopTracks } from '../lib/spotifyApi'

const SUGGESTIONS = [
  "What does my music taste say about me?",
  "Recommend something energetic",
  "Give me 3 tracks for a chill evening",
  "What makes a track sound calm vs energetic?",
]

// When the assistant cites Spotify tracks, the frontend renders them as
// cards (album art + title + artist + link). The text bubble is therefore
// noisy if it still carries the raw markdown link too — collapse each
// `[Track — Artist](https://open.spotify.com/track/…)` link to its visible
// label so the bubble reads as plain prose and the card supplies the link.
function stripSpotifyLinks(text) {
  if (!text) return text
  return text.replace(
    /\[([^\]]+)\]\(https?:\/\/open\.spotify\.com\/track\/[^\)]+\)/g,
    '$1',
  )
}

/**
 * Heart button on a chat track card. Writes to the Auralis profile via
 * /spotify/feedback (Spotify Liked Songs is gated behind the production
 * approval flag, same pattern as SpotifyRail / SpotifyTasteRail). Pre-
 * filled when the track is already in the user's interaction_log.
 *
 * The card itself is the link to Spotify, so the heart button stops
 * click propagation to avoid triggering the Spotify-open while the user
 * is just trying to like.
 */
function ChatTrackHeart({ track, savedMap, onLiked, onFeedback }) {
  const initiallySaved = !!(savedMap && savedMap[track.spotify_id])
  const [saved, setSaved] = useState(initiallySaved)
  const [busy, setBusy]   = useState(false)
  useEffect(() => { setSaved(initiallySaved) }, [initiallySaved])

  if (!track?.spotify_id) return null

  const onClick = async (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (busy || saved) return
    setBusy(true)
    setSaved(true) // optimistic
    try {
      await axios.post(`${API}/spotify/feedback`, {
        spotify_id: track.spotify_id,
        track_name: track.name,
        artists:    track.artists || [],
        label:      'like',
      })
      if (typeof onLiked === 'function') onLiked(track.spotify_id)
      if (typeof onFeedback === 'function') {
        try { onFeedback() } catch { /* swallow — like already saved */ }
      }
    } catch {
      setSaved(false) // rollback
    } finally {
      setBusy(false)
    }
  }

  return (
    <button
      className={`chat-track-heart ${saved ? 'saved' : ''}`}
      onClick={onClick}
      disabled={busy || saved}
      aria-label={saved ? 'Liked' : 'Like'}
      title={saved ? 'Liked — recorded in your Auralis profile' : 'Like (records to Auralis profile)'}
    >
      {saved ? '♥' : '♡'}
    </button>
  )
}


export default function Chat({ onFeedback = null } = {}) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [spotifyContext, setSpotifyContext] = useState(null)
  // savedMap[spotify_id] = boolean — populated from /profile/spotify-likes so
  // recommended tracks the user has previously hearted render with a filled
  // heart from the moment the card appears, not just after a fresh click.
  const [savedMap, setSavedMap] = useState({})
  const bottomRef = useRef()
  const { isLoggedIn, user } = useSpotifyAuth()

  // Pull the Auralis-side likes once on mount so the chat heart state
  // reflects what's already in the user's profile.
  useEffect(() => {
    let cancelled = false
    axios
      .get(`${API}/profile/spotify-likes`)
      .then((res) => {
        if (cancelled) return
        const next = {}
        for (const id of (res.data?.liked || [])) next[id] = true
        setSavedMap(next)
      })
      .catch(() => { /* default to empty — hearts just start unfilled */ })
    return () => { cancelled = true }
  }, [])

  const markLocallyLiked = (spotifyId) => {
    setSavedMap((prev) => (prev[spotifyId] ? prev : { ...prev, [spotifyId]: true }))
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Pre-fetch the user's Spotify listening highlights once per session so we
  // can ground taste-related answers in real play history. The token never
  // leaves the browser — we only forward de-personalised highlights to the
  // backend (artist names + track names + the genre tags Spotify attaches
  // to each top artist), not the access token itself. Genres are critical
  // for artist-style follow-ups: without them, "Dave-style" queries can
  // surface country tracks that share a mood keyword but no genre context.
  useEffect(() => {
    if (!isLoggedIn) { setSpotifyContext(null); return }
    let cancelled = false
    ;(async () => {
      try {
        const [artistsResp, tracksResp] = await Promise.all([
          getTopArtists({ timeRange: 'medium_term', limit: 8 }),
          getTopTracks({ timeRange: 'medium_term', limit: 10 }),
        ])
        if (cancelled) return

        const artistItems = artistsResp.items || []
        // Dedupe the genre list across the top artists so a 3-artist
        // hip-hop dominance doesn't crowd out genres from a 4th artist
        // in a different bucket. Cap at 6 to keep the candidate-fetch
        // query strings short.
        const seenGenres = new Set()
        const topGenres = []
        for (const a of artistItems) {
          for (const g of (a?.genres || [])) {
            if (!seenGenres.has(g)) {
              seenGenres.add(g)
              topGenres.push(g)
            }
          }
          if (topGenres.length >= 6) break
        }

        // Per-artist record — name, Spotify ID, and genre tags. The ID
        // is critical for the chat backend: when the user says "X-style"
        // and X is one of these top artists, the backend uses the ID to
        // call /artists/{id}/top-tracks for VERIFIED tracks by that
        // exact artist, sidestepping the artist:"X" search ambiguity
        // (e.g. Dave Matthews vs Dave the UK rapper both match "Dave").
        const artistGenres = artistItems.map((a) => ({
          name:        a?.name || '',
          spotify_id:  a?.id   || '',
          genres:      (a?.genres || []).slice(0, 4),
        })).filter((a) => a.name)

        setSpotifyContext({
          display_name: user?.display_name || user?.id || null,
          top_artists: artistItems.map((a) => a?.name).filter(Boolean),
          top_genres: topGenres,
          artist_genres: artistGenres,
          top_tracks: (tracksResp.items || []).map((t) => ({
            name: t.name,
            artists: (t.artists || []).map((a) => a.name),
          })),
        })
      } catch {
        if (!cancelled) setSpotifyContext(null)
      }
    })()
    return () => { cancelled = true }
  }, [isLoggedIn, user?.id])

  const send = async (text) => {
    if (!text.trim()) return
    const userMsg = { role: 'user', content: text }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const res = await axios.post(`${API}/chat`, {
        messages: newMessages.map(m => ({ role: m.role, content: m.content })),
        spotify_context: spotifyContext,
      })
      setMessages([
        ...newMessages,
        {
          role: 'assistant',
          content: res.data.response,
          tracks: res.data.tracks || [],
        },
      ])
    } catch (e) {
      const detail = e.response?.data?.detail
      let errMsg = 'Sorry, something went wrong.'
      if (typeof detail === 'string' && detail.includes('credit')) {
        errMsg = '⚠️ API credits needed. Add credits at console.anthropic.com to use the chat.'
      } else if (detail) {
        errMsg = `Error: ${detail}`
      }
      setMessages([...newMessages, { role: 'assistant', content: errMsg }])
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) }
  }

  return (
    <div className="chat-container">
      <div style={{ marginBottom: 24 }}>
        <h1 className="page-title">Chat</h1>
        <p className="page-subtitle">Ask about your music taste, get recommendations, or explore acoustic features.</p>
      </div>

      {messages.length === 0 && (
        <div className="chat-suggestions">
          {SUGGESTIONS.map(s => (
            <button key={s} className="suggestion-chip" onClick={() => send(s)}>{s}</button>
          ))}
        </div>
      )}

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="chat-avatar">
              {msg.role === 'user' ? '◉' : '◈'}
            </div>
            <div className="chat-bubble">
              {/* For assistant messages we ALWAYS strip Spotify URLs from
                  the prose. Two reasons:
                    (1) Tracks the backend matched against the candidate
                        pool render as cards below — the in-prose link is
                        redundant.
                    (2) When the LLM hallucinates a URL for a track NOT
                        in the pool (e.g. recycling Marvellous/Selfish
                        from training data), the card layer correctly
                        drops it but the raw markdown link would still
                        appear in the bubble. Stripping unconditionally
                        kills that leak — what's left is the prose
                        framing + cards for whatever actually validated. */}
              {msg.role === 'assistant'
                ? stripSpotifyLinks(msg.content)
                : msg.content}

              {msg.role === 'assistant' && msg.tracks?.length > 0 && (
                <div className="chat-track-cards">
                  {msg.tracks.map((t) => (
                    <a
                      key={t.spotify_id}
                      className="chat-track-card"
                      href={t.external_url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <div className="chat-track-card-art-wrap">
                        {t.album_art_url ? (
                          <img src={t.album_art_url} alt={t.album || ''} loading="lazy" />
                        ) : (
                          <div className="chat-track-card-art-fallback">♪</div>
                        )}
                        <ChatTrackHeart
                          track={t}
                          savedMap={savedMap}
                          onLiked={markLocallyLiked}
                          onFeedback={onFeedback}
                        />
                      </div>
                      <div className="chat-track-card-meta">
                        <div className="chat-track-card-name" title={t.name}>{t.name}</div>
                        <div className="chat-track-card-artists" title={(t.artists || []).join(', ')}>
                          {(t.artists || []).join(', ') || 'Unknown artist'}
                        </div>
                        <div className="chat-track-card-link">Open in Spotify ↗</div>
                      </div>
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        {loading && (
          <div className="chat-message assistant">
            <div className="chat-avatar">◈</div>
            <div className="chat-bubble">
              <div className="loading" style={{ padding: 0 }}>
                <div className="spinner" /> Thinking...
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          placeholder="Ask Auralis anything about your music..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
        />
        <button className="btn btn-primary" onClick={() => send(input)} disabled={!input.trim() || loading}>
          ↑
        </button>
        {messages.length > 0 && (
          <button className="btn btn-ghost" onClick={() => setMessages([])} title="Clear chat">✕</button>
        )}
      </div>
    </div>
  )
}
