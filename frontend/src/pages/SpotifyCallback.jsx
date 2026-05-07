/**
 * Landing page for Spotify's OAuth redirect.
 *
 * Spotify sends the browser to ${origin}/callback?code=...&state=... after the
 * user authorises us. This page:
 *   1. Reads `code` and `state` from the URL
 *   2. Calls completeLogin() to exchange them for tokens (PKCE, no secret)
 *   3. Redirects back to /profile so the user lands on a page that shows
 *      their newly-connected status
 *
 * If Spotify returned an error (user denied access, scope mismatch, etc.) we
 * surface the message and offer a "Try again" button instead of silently
 * failing.
 */

import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { completeLogin, beginLogin } from '../lib/spotifyAuth'

export default function SpotifyCallback() {
  const [params]   = useSearchParams()
  const navigate   = useNavigate()
  const [status, setStatus] = useState('exchanging')  // 'exchanging' | 'error'
  const [error,  setError]  = useState(null)

  useEffect(() => {
    const code  = params.get('code')
    const state = params.get('state')
    const err   = params.get('error')

    if (err) {
      setStatus('error')
      setError(`Spotify returned an error: ${err}`)
      return
    }
    if (!code || !state) {
      setStatus('error')
      setError('Missing code or state in callback URL.')
      return
    }

    completeLogin({ code, state })
      .then(() => {
        // Land the user on Profile so they immediately see the connection.
        navigate('/profile', { replace: true })
      })
      .catch((e) => {
        setStatus('error')
        setError(e.message || String(e))
      })
  }, [params, navigate])

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Connecting to Spotify…</h1>
      </div>

      {status === 'exchanging' && (
        <div className="loading"><div className="spinner" /> Finishing sign-in…</div>
      )}

      {status === 'error' && (
        <div className="empty-state">
          <div className="empty-state-icon">⚠</div>
          <div className="empty-state-title">Couldn't complete sign-in</div>
          <div className="empty-state-text" style={{ marginBottom: 20 }}>{error}</div>
          <button className="btn btn-primary" onClick={() => beginLogin()}>
            Try again
          </button>
        </div>
      )}
    </div>
  )
}
