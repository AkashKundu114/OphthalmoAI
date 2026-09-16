// Web Audio API procedural sound synthesizer inspired by uisfx.com
// Zero external audio files or network requests required.

let audioCtx = null

function getAudioContext() {
  if (typeof window === 'undefined') return null
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (AudioContextClass) {
      audioCtx = new AudioContextClass()
    }
  }
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume().catch(() => {})
  }
  return audioCtx
}

export function isSoundEnabled() {
  if (typeof window === 'undefined') return false
  const saved = localStorage.getItem('ophthalmo_sfx_enabled')
  return saved === null ? true : saved === 'true'
}

export function setSoundEnabled(enabled) {
  if (typeof window === 'undefined') return
  localStorage.setItem('ophthalmo_sfx_enabled', String(enabled))
  if (enabled) {
    playToggleSound()
  }
}

/**
 * Tactile micro-click sound (inspired by evilbuttons and tactile UI)
 */
export function playClickSound() {
  if (!isSoundEnabled()) return
  try {
    const ctx = getAudioContext()
    if (!ctx) return
    const now = ctx.currentTime

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.type = 'sine'
    osc.frequency.setValueAtTime(320, now)
    osc.frequency.exponentialRampToValueAtTime(80, now + 0.04)

    gain.gain.setValueAtTime(0.08, now)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04)

    osc.connect(gain)
    gain.connect(ctx.destination)

    osc.start(now)
    osc.stop(now + 0.04)
  } catch {
    // Ignore audio errors gracefully
  }
}

/**
 * Medical laser / radar sweep sound when scanning initiates
 */
export function playScanStartSound() {
  if (!isSoundEnabled()) return
  try {
    const ctx = getAudioContext()
    if (!ctx) return
    const now = ctx.currentTime

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.type = 'sine'
    osc.frequency.setValueAtTime(260, now)
    osc.frequency.exponentialRampToValueAtTime(680, now + 0.28)

    gain.gain.setValueAtTime(0.01, now)
    gain.gain.linearRampToValueAtTime(0.06, now + 0.08)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.32)

    osc.connect(gain)
    gain.connect(ctx.destination)

    osc.start(now)
    osc.stop(now + 0.32)
  } catch {
    // Ignore audio errors gracefully
  }
}

/**
 * Crystal medical diagnostic success chime (C5 -> E5 -> G5 harmonic triad)
 */
export function playSuccessChime() {
  if (!isSoundEnabled()) return
  try {
    const ctx = getAudioContext()
    if (!ctx) return
    const now = ctx.currentTime
    const notes = [523.25, 659.25, 783.99] // C5, E5, G5

    notes.forEach((freq, index) => {
      const startTime = now + index * 0.07
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()

      osc.type = 'sine'
      osc.frequency.setValueAtTime(freq, startTime)

      gain.gain.setValueAtTime(0.06, startTime)
      gain.gain.exponentialRampToValueAtTime(0.0008, startTime + 0.35)

      osc.connect(gain)
      gain.connect(ctx.destination)

      osc.start(startTime)
      osc.stop(startTime + 0.35)
    })
  } catch {
    // Ignore audio errors gracefully
  }
}

/**
 * Micro toggle pop
 */
export function playToggleSound() {
  if (!isSoundEnabled()) return
  try {
    const ctx = getAudioContext()
    if (!ctx) return
    const now = ctx.currentTime

    const osc = ctx.createOscillator()
    const gain = ctx.createGain()

    osc.type = 'triangle'
    osc.frequency.setValueAtTime(440, now)
    osc.frequency.exponentialRampToValueAtTime(880, now + 0.05)

    gain.gain.setValueAtTime(0.05, now)
    gain.gain.exponentialRampToValueAtTime(0.001, now + 0.05)

    osc.connect(gain)
    gain.connect(ctx.destination)

    osc.start(now)
    osc.stop(now + 0.05)
  } catch {
    // Ignore audio errors gracefully
  }
}
