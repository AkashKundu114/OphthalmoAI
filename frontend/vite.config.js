import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
    const allowedHosts = (process.env.VITE_ALLOWED_HOSTS || '')
    .split(',')
    .map(h => h.trim())
    .filter(Boolean)

  const backendTarget = process.env.VITE_DEV_BACKEND_URL || 'http://localhost:8000';

  return {
    plugins: [react()],
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: './tests/setup.js'
    },
    server: {
      ...(allowedHosts.length > 0 ? { allowedHosts } : {}),
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
          secure: false,
        }
      }
    }
  }
})