import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const allowedHosts = (env.VITE_ALLOWED_HOSTS || process.env.VITE_ALLOWED_HOSTS || '')
    .split(',')
    .map(h => h.trim())
    .filter(Boolean)

  const backendTarget = env.VITE_DEV_BACKEND_URL || process.env.VITE_DEV_BACKEND_URL || 'http://localhost:8000';
  const apiUrl = env.VITE_API_URL || env.API_URL || env.BACKEND_URL || process.env.VITE_API_URL || process.env.API_URL || process.env.BACKEND_URL || '';

  return {
    define: apiUrl ? {
      'import.meta.env.VITE_API_URL': JSON.stringify(apiUrl),
      'import.meta.env.API_URL': JSON.stringify(apiUrl),
    } : {},
    envPrefix: ['VITE_', 'API_'],
    plugins: [react()],
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: './tests/setup.js'
    },
    build: {
      chunkSizeWarningLimit: 1600,
    },
    server: {
      port: 5176,
      ...(allowedHosts.length > 0 ? { allowedHosts } : {}),
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
          secure: false,
          rewrite: (path) => path.replace(/^\/api/, ''),
        }
      }
    }
  }
})