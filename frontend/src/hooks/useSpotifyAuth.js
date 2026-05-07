/**
 * React hook that exposes the Spotify auth state to components.
 *
 * What it provides
 * ----------------
 * - `user`         the logged-in Spotify profile (id, display_name, email, images),
 *                  or null if not connected
 * - `isLoggedIn`   convenience boolean
 * - `loading`      true on initial mount while we fetch /me
 * - `login()`      kick off the PKCE redirect to Spotify
 * - `logout()`     clear local tokens (does NOT revoke on Spotify's side; users
 *                  can do that at https://www.spotify.com/account/apps if they
 *                  want a fully clean break)
 * - `refreshUser()` re-fetch the profile, e.g. after the callback page returns
 *
 * Why a hook
 * ----------
 * Multiple pages need to know whether the user is connected (Profile to show
 * status, SpotifyRail to render a "+ Save" button, Playlist to enable
 * "Push to Spotify"). Centralising the state here keeps the access pattern
 * consistent and ensures we only hit /me once per mount per component.
 */

import { useEffect, useState, useCallback } from 'react'
import { beginLogin, logout as clearTokens, isLoggedIn as hasTokens } from '../lib/spotifyAuth'
import { getCurrentUser } from '../lib/spotifyApi'

export function useSpotifyAuth() {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const refreshUser = useCallback(async () => {
    if (!hasTokens()) {
      setUser(null)
      setLoading(false)
      return
    }
    try {
      const me = await getCurrentUser()
      setUser(me)
      setError(null)
    } catch (e) {
      // If /me fails the token is probably revoked. Clear and treat as logged out.
      setUser(null)
      setError(e.message)
      clearTokens()
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refreshUser() }, [refreshUser])

  const login = useCallback(async () => {
    await beginLogin()  // navigates away — no further code runs
  }, [])

  const logout = useCallback(() => {
    clearTokens()
    setUser(null)
  }, [])

  return {
    user,
    isLoggedIn: !!user,
    loading,
    error,
    login,
    logout,
    refreshUser,
  }
}
