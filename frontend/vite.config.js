import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
//
// We pin the dev server host to 127.0.0.1 (IPv4 loopback) on purpose. By
// default Vite binds to "localhost", which on modern Windows resolves to the
// IPv6 loopback (::1) only — so http://127.0.0.1:5173 fails with
// ERR_CONNECTION_REFUSED. That breaks the Spotify PKCE flow because Spotify's
// dashboard accepts the loopback IP literal but not the "localhost" hostname,
// so the registered redirect URI is http://127.0.0.1:5173/callback.
//
// Forcing host to 127.0.0.1 here means both the user's browser tab and the
// Spotify redirect land on the same address that Vite is actually listening
// on, with no fiddly hosts-file workarounds.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
  },
})
