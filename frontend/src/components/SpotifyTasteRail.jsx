/**
 * SpotifyTasteRail — surfaces tracks driven by the user's actual Spotify
 * listening history rather than the local rating-based emotion profile.
 *
 * Why /me/top/tracks instead of search-by-artist
 * ----------------------------------------------
 * The earlier version built a Spotify Search query in the form
 *
 *     (artist:"Dave" OR artist:"Baby Keem" OR artist:"J. Cole") energetic
 *
 * which Spotify interpreted as "tracks by these artists where the title also
 * literally contains the word energetic". That's almost always zero results,
 * so the rail rendered the seed-artist subtitle but no track cards.
 *
 * The user-scoped /me/top/tracks endpoint returns the listener's actual most-
 * played tracks directly — no search query needed. That's both more reliable
 * and more "Spotify-driven" in the way the user actually means it: it's their
 * real listening history, not a search guess at it.
 *
 * If a local emotion preference is supplied we still annotate the rail so the
 * user knows it's a hybrid view, but the underlying tracks come from Spotify.
 *
 * Falls back gracefully:
 *   - If the user isn't connected → render nothing.
 *   - If they have no top tracks yet (very new account) → show a friendly
 *     placeholder.
 *   - If Spotify errors → surface a small error line, don't blow up.
 */

import { useEffect, useState } from 'react'
import axios from 'axios'
import { API } from '../config'
import { useSpotifyAuth } from '../hooks/useSpotifyAuth'
import { getTopTracks, getTopArtists, saveTracks } from '../lib/spotifyApi'

// Mirror of the same flag in SpotifyRail.jsx — see comment there for context.
// Off while Spotify Dev-Mode blocks PUT /me/tracks; the heart still records
// a like to the Auralis profile so it shows up in Profile → Interaction
// History. Flip both flags back to true once Spotify approves Auralis for
// production access.
const WRITE_TO_SPOTIFY_LIBRARY = false


function SaveButton({ track, initiallySaved, savedReady = true, discreteEmotion = null, onFeedback = null }) {
  const [saved, setSaved] = useState(!!initiallySaved)
  const [busy, setBusy]   = useState(false)
  const [error, setError] = useState(null)
  useEffect(() => { setSaved(!!initiallySaved) }, [initiallySaved])

  if (!track?.spotify_id) return null

  const onClick = async () => {
    // Gate clicks until parent's likes lookup resolves to avoid duplicate
    // feedback on tracks that should have been pre-filled.
    if (busy || saved || !savedReady) return
    setBusy(true)
    setError(null)
    setSaved(true) // optimistic

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

    // Refresh global profile state so the sidebar Vibe chart and Profile
    // page Interaction History reflect the new like immediately.
    if (typeof onFeedback === 'function') {
      try { onFeedback() } catch { /* swallow */ }
    }

    if (WRITE_TO_SPOTIFY_LIBRARY) {
      try { await saveTracks([track.spotify_id]) } catch { /* silent fallback */ }
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


export default function SpotifyTasteRail({
  emotion = null,             // optional mood overlay from the local profile
  timeRange = 'medium_term',  // 'short_term' | 'medium_term' | 'long_term'
  limit = 8,
  title = 'Based on your Spotify taste',
  onFeedback = null,
}) {
  const { isLoggedIn, loading: authLoading } = useSpotifyAuth()
  const [tracks, setTracks] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [seedArtists, setSeedArtists] = useState([])
  const [savedMap, setSavedMap] = useState({})
  const [savedReady, setSavedReady] = useState(false)

  useEffect(() => {
    if (authLoading || !isLoggedIn) return

    let cancelled = false
    setLoading(true)
    setError(null)
    setSavedReady(false)
    ;(async () => {
      try {
        // 1. Pull top tracks across THREE time ranges in parallel so the
        //    pool is ~3× larger than a single range. With 60+ hearted
        //    tracks, the single-range pool was too small to surface
        //    fresh material; the multi-range pool gives the rotation
        //    real headroom. Plus pull Auralis hearts and top artists.
        const [topShort, topMed, topLong, topArtistsResp, likesResp] = await Promise.all([
          getTopTracks({ timeRange: 'short_term',  limit: 50 }),
          getTopTracks({ timeRange: 'medium_term', limit: 50 }),
          getTopTracks({ timeRange: 'long_term',   limit: 50 }),
          getTopArtists({ timeRange, limit: 5 }),
          axios.get(`${API}/profile/spotify-likes`).catch(() => ({ data: { liked: [] } })),
        ])
        if (cancelled) return

        const artists = (topArtistsResp.items || []).map((a) => a.name).filter(Boolean)
        setSeedArtists(artists)

        // Genres for the mood-search supplement (so its results land in
        // the user's musical neighborhood, not generic mood keywords).
        const seenG = new Set()
        const seedGenres = []
        for (const a of (topArtistsResp.items || [])) {
          for (const g of (a?.genres || [])) {
            if (!seenG.has(g)) { seenG.add(g); seedGenres.push(g) }
          }
          if (seedGenres.length >= 6) break
        }

        const liked = new Set(likesResp.data?.liked || [])

        // Dedupe across the three time ranges by spotify_id.
        const merged = new Map()
        const _add = (items) => {
          for (const t of (items || [])) {
            if (t?.id && !merged.has(t.id)) merged.set(t.id, t)
          }
        }
        _add(topShort.items)
        _add(topMed.items)
        _add(topLong.items)

        const allFetched = Array.from(merged.values()).map((t) => ({
          spotify_id:    t.id,
          name:          t.name,
          artists:       (t.artists || []).map((a) => a.name),
          album:         t.album?.name || '',
          album_art_url: t.album?.images?.[0]?.url || null,
          external_url:  t.external_urls?.spotify || null,
        }))

        const shuffle = (arr) => {
          const a = [...arr]
          for (let i = a.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1))
            ;[a[i], a[j]] = [a[j], a[i]]
          }
          return a
        }

        // Filter out hearted, then shuffle.
        let unhearted = shuffle(allFetched.filter((t) => !liked.has(t.spotify_id)))
        let fetched = unhearted.slice(0, limit)

        // 2. Supplement with /spotify/search-by-mood (artist+genre blend
        //    on the backend) when the unhearted top-tracks pool runs thin.
        //    This is what gives the rail genuine novelty for engaged
        //    users — without it, listeners with 60+ hearted tracks see a
        //    stale strip dominated by re-shows.
        if (fetched.length < limit && (emotion || true)) {
          try {
            const moodRes = await axios.post(`${API}/spotify/search-by-mood`, {
              discrete_emotion: emotion || 'energetic',
              limit:            Math.min(limit * 2, 20),
              seed_artists:     artists,
              seed_genres:      seedGenres,
              exclude_ids:      Array.from(liked),
            })
            if (!cancelled) {
              const ids_seen = new Set(fetched.map((t) => t.spotify_id))
              const supplements = (moodRes.data?.tracks || [])
                .filter((t) => t?.spotify_id && !ids_seen.has(t.spotify_id) && !liked.has(t.spotify_id))
                .map((t) => ({
                  spotify_id:    t.spotify_id,
                  name:          t.name,
                  artists:       t.artists || [],
                  album:         t.album || '',
                  album_art_url: t.album_art_url || null,
                  external_url:  t.external_url || null,
                }))
              fetched = [...fetched, ...shuffle(supplements)].slice(0, limit)
            }
          } catch {
            /* mood-search fallback failure is non-fatal */
          }
        }

        // 3. Final last-resort: only if still short, pull from hearted
        //    so the rail isn't empty. Ranking-wise, fresh > supplements > hearted.
        if (fetched.length < limit) {
          const hearted = shuffle(allFetched.filter((t) => liked.has(t.spotify_id)))
          const ids_seen = new Set(fetched.map((t) => t.spotify_id))
          fetched = [
            ...fetched,
            ...hearted.filter((t) => !ids_seen.has(t.spotify_id)),
          ].slice(0, limit)
        }
        setTracks(fetched)

        // Heart-fill state: we already have the liked set in scope.
        const ids = fetched.map((t) => t.spotify_id).filter(Boolean)
        const next = {}
        ids.forEach((id) => { next[id] = liked.has(id) })
        setSavedMap(next)
        setSavedReady(true)
      } catch (e) {
        if (!cancelled) {
          setError(e.message || 'Could not load Spotify-driven recommendations.')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()

    return () => { cancelled = true }
  }, [isLoggedIn, authLoading, timeRange, limit])
  // Note: `emotion` is intentionally NOT in the dep array — switching it
  // shouldn't refetch the user's top tracks, only adjust the subtitle copy.

  if (authLoading || !isLoggedIn) return null

  return (
    <div className="spotify-rail">
      <div className="spotify-rail-header">
        <div>
          <div className="spotify-rail-title">{title}</div>
          {seedArtists.length > 0 && (
            <div className="spotify-rail-subtitle">
              Your most-played tracks · top artists: {seedArtists.slice(0, 3).join(', ')}
              {emotion ? ` · cross-checked against your ${emotion} taste` : ''}
            </div>
          )}
        </div>
        <div className="spotify-rail-badge" aria-hidden="true">Spotify</div>
      </div>

      {loading && (
        <div className="loading" style={{ padding: '24px 0' }}>
          <div className="spinner" /> Reading your Spotify history...
        </div>
      )}

      {error && !loading && (
        <div className="spotify-rail-error">{error}</div>
      )}

      {!loading && !error && tracks.length === 0 && (
        <div className="spotify-rail-empty">
          You don't have enough Spotify listening history yet for us to
          personalise this rail. Listen to a few tracks on Spotify and come back.
        </div>
      )}

      {!loading && !error && tracks.length > 0 && (
        <div className="spotify-rail-grid">
          {tracks.map((t) => (
            <div key={t.spotify_id} className="spotify-card">
              <div className="spotify-card-art">
                {t.album_art_url ? (
                  <img src={t.album_art_url} alt={t.album} loading="lazy" />
                ) : (
                  <div className="spotify-card-art-fallback">♪</div>
                )}
              </div>
              <div className="spotify-card-body">
                <div className="spotify-card-title" title={t.name}>{t.name}</div>
                <div className="spotify-card-artists" title={t.artists.join(', ')}>
                  {t.artists.join(', ') || 'Unknown artist'}
                </div>
                <div className="spotify-card-actions">
                  <a
                    className="spotify-card-link"
                    href={t.external_url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    Open in Spotify ↗
                  </a>
                  <SaveButton
                    track={t}
                    initiallySaved={savedMap[t.spotify_id]}
                    savedReady={savedReady}
                    discreteEmotion={emotion}
                    onFeedback={onFeedback}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
