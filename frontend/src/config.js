/**
 * Auralis frontend config.
 *
 * Central API base URL. In development Vite serves on 5173 and the backend
 * runs on localhost:8000, so the default works out of the box. For production
 * (Vercel), set `VITE_API_URL` in the Vercel project settings to the Fly.io
 * backend URL, e.g. `https://auralis.fly.dev`. Vite inlines the value at
 * build time, so every API call in the bundle hits the right host.
 */
export const API =
  (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/+$/, '')
