import React, { useState, useRef, useCallback, useEffect } from 'react'
import { Eye, Flame, MoveHorizontal, Maximize2 } from 'lucide-react'
import { playClickSound } from '../utils/soundEffects'

/**
 * SplitSenseSlider - Interactive before/after split comparison slider inspired by splitsense.ai
 * Allows seamless side-by-side or sliding comparison of original retinal scan vs AI Grad-CAM lesion heatmap.
 */
export default function SplitSenseSlider({ originalImage, heatmapImage, alt = 'Retinal Scan Comparison' }) {
  const [sliderPos, setSliderPos] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef(null)

  const handleMove = useCallback((clientX) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPos(percent)
  }, [])

  const onMouseDown = () => {
    setIsDragging(true)
    playClickSound()
  }

  const onTouchStart = () => {
    setIsDragging(true)
    playClickSound()
  }

  useEffect(() => {
    const onMouseMove = (e) => {
      if (!isDragging) return
      handleMove(e.clientX)
    }

    const onMouseUp = () => {
      if (isDragging) setIsDragging(false)
    }

    const onTouchMove = (e) => {
      if (!isDragging || !e.touches[0]) return
      handleMove(e.touches[0].clientX)
    }

    const onTouchEnd = () => {
      if (isDragging) setIsDragging(false)
    }

    if (isDragging) {
      window.addEventListener('mousemove', onMouseMove)
      window.addEventListener('mouseup', onMouseUp)
      window.addEventListener('touchmove', onTouchMove)
      window.addEventListener('touchend', onTouchEnd)
    }

    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('touchend', onTouchEnd)
    }
  }, [isDragging, handleMove])

  // If only one image is available, fallback to single image viewer
  if (!heatmapImage) {
    return (
      <div className="relative rounded-2xl overflow-hidden border border-slate-200 shadow-sm bg-slate-900 aspect-square max-w-md mx-auto flex items-center justify-center">
        <img src={originalImage} alt={alt} className="w-full h-full object-contain" />
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {/* Control / Legend Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs px-0.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 font-semibold text-[11px] shadow-2xs">
            <Eye className="w-3.5 h-3.5 text-cyan-600 shrink-0" />
            Original Image
          </span>
          <span className="text-slate-400 font-bold text-[11px]">vs</span>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-50 border border-rose-200 text-rose-700 font-semibold text-[11px] shadow-2xs">
            <Flame className="w-3.5 h-3.5 text-rose-500 shrink-0" />
            AI Attention Heatmap
          </span>
        </div>
        <div className="inline-flex items-center px-2 py-0.5 rounded-md bg-slate-100 border border-slate-200 text-[11px] font-mono font-semibold text-slate-600 shadow-2xs shrink-0">
          Split: {Math.round(sliderPos)}%
        </div>
      </div>

      {/* Interactive Split Frame */}
      <div
        ref={containerRef}
        onMouseDown={onMouseDown}
        onTouchStart={onTouchStart}
        className="relative aspect-square max-w-md mx-auto rounded-2xl overflow-hidden border-2 border-slate-200 shadow-md bg-slate-950 select-none cursor-ew-resize group"
      >
        {/* Heatmap Layer (Full Background) */}
        <img
          src={heatmapImage}
          alt="AI Attention Heatmap Overlay"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Original Image Layer (Clipped to Slider Percentage) */}
        <div
          className="absolute inset-0 overflow-hidden pointer-events-none"
          style={{ width: `${sliderPos}%` }}
        >
          <img
            src={originalImage}
            alt="Original Retinal Scan"
            className="absolute inset-0 w-full h-full object-contain max-w-none"
            style={{ width: containerRef.current ? `${containerRef.current.clientWidth}px` : '100%' }}
          />
        </div>

        {/* Vertical Divider Line with Specular Highlight */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-white shadow-[0_0_12px_rgba(0,173,181,0.8)] pointer-events-none"
          style={{ left: `${sliderPos}%` }}
        >
          {/* Circular Tactile Thumb (inspired by dialkit/evilbuttons) */}
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-white border-2 border-cyan-500 shadow-lg flex items-center justify-center text-cyan-600 transition-transform group-hover:scale-110">
            <MoveHorizontal className="w-4 h-4" />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-center gap-2 text-[11px] text-slate-500 text-center">
        <span>Drag the slider handle to inspect where the AI detected pathological features</span>
      </div>
    </div>
  )
}
