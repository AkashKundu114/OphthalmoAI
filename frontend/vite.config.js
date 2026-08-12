import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
    const allowedHosts = (process.env.VITE_ALLOWED_HOSTS || '')
    .split(',')
    .map(h => h.trim())
    .filter(Boolean)

  /**
   * Backend target for the /api proxy in development.
   * Override in frontend/.env.local:
   *   VITE_DEV_BACKEND_URL=http://localhost:8000
   */
  const backendTarget = process.env.VITE_DEV_BACKEND_URL || 'http:

  return {
    plugins: [react()],

    server: {
      
      ...(allowedHosts.length > 0 ? { allowedHosts } : {}),

      proxy: {
        