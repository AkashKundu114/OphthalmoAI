export const FALLBACK_TUNNEL_URL = 'https://preventing-eur-able-stuff.trycloudflare.com'
export const VERCEL_API_URL = 'https://ophthalmo-ai-mu.vercel.app/api'

export const getActiveApiUrl = () => {
  if (typeof window !== 'undefined') {
    const custom = window.localStorage?.getItem('ophthalmo_api_url')
    if (custom && custom.trim()) {
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
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, '')
  }

  if (FALLBACK_TUNNEL_URL && FALLBACK_TUNNEL_URL.trim()) {
    return FALLBACK_TUNNEL_URL.trim().replace(/\/+$/, '')
  }

  if (typeof window !== 'undefined' && window.location.hostname.includes('hf.space')) {
    return VERCEL_API_URL
  }

  return '/api'
}

