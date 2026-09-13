export const FALLBACK_TUNNEL_URL = ''

export const getActiveApiUrl = () => {
  if (typeof window !== 'undefined') {
    const custom = window.localStorage?.getItem('ophthalmo_api_url')
    if (custom && custom.trim() && !custom.includes('trycloudflare.com')) {
      return custom.trim().replace(/\/+$/, '')
    }
    const hostname = window.location.hostname
    if (
      hostname === 'localhost' ||
      hostname === '127.0.0.1' ||
      hostname === '0.0.0.0' ||
      hostname.endsWith('.local') ||
      hostname.includes('vercel.app')
    ) {
      return '/api'
    }
  }
  const envUrl = import.meta.env.VITE_API_URL || import.meta.env.API_URL
  if (envUrl && envUrl.trim() && !envUrl.includes('trycloudflare.com')) {
    return envUrl.trim().replace(/\/+$/, '')
  }
  return '/api'
}
