import React, { useEffect, useState } from 'react'
import { Scan, Eye, Activity } from 'lucide-react'

/**
 * RetinaScanShader - Cyber-medical laser scan overlay inspired by shaders.com & bookofshapes.com
 * Renders during inference with a sweeping laser line, target reticle crosshairs, and live coordinate HUD.
 */
export default function RetinaScanShader({ stage = 'Analyzing Fundus Biomarkers...' }) {
  const [scanCoord, setScanCoord] = useState({ x: 142, y: 218 })

  useEffect(() => {
    const interval = setInterval(() => {
      setScanCoord({
        x: Math.floor(100 + Math.random() * 200),
        y: Math.floor(80 + Math.random() * 240),
      })
    }, 400)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="absolute inset-0 z-30 pointer-events-none rounded-2xl overflow-hidden backdrop-blur-[1px] bg-cyan-950/20 flex flex-col justify-between p-4 border-2 border-cyan-400/50 shadow-[inset_0_0_24px_rgba(0,173,181,0.25)]">
      {/* Top HUD Telemetry */}
      <div className="flex items-center justify-between text-[11px] font-mono text-cyan-300 font-bold tracking-wider">
        <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/40">
          <Activity className="w-3 h-3 text-cyan-400 animate-spin" />
          NEURAL SCAN ACTIVE
        </span>
        <span className="px-2 py-0.5 rounded-full bg-cyan-950/80 border border-cyan-500/40 tabular-nums">
          LOC [{scanCoord.x}:{scanCoord.y}]
        </span>
      </div>

      {/* Sweeping Laser Beam (shaders.com simulation via CSS animation) */}
      <div 
        className="absolute left-0 right-0 h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_16px_#00ADB5,0_0_32px_#00ADB5] animate-laser-sweep"
        style={{
          animation: 'laserSweep 2.2s ease-in-out infinite alternate',
        }}
      />

      {/* Center Target Reticle (bookofshapes.com) */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-none">
        <div className="relative w-24 h-24 rounded-full border border-cyan-400/40 flex items-center justify-center animate-pulse">
          <div className="w-12 h-12 rounded-full border border-cyan-300/60 flex items-center justify-center">
            <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_8px_#00ADB5]" />
          </div>
          {/* Crosshairs */}
          <div className="absolute top-0 bottom-0 w-px bg-cyan-400/30" />
          <div className="absolute left-0 right-0 h-px bg-cyan-400/30" />
        </div>
      </div>

      {/* Bottom Status Pill */}
      <div className="flex items-center justify-center">
        <div className="px-3 py-1 rounded-full bg-slate-900/90 border border-cyan-500/50 text-cyan-300 text-xs font-semibold flex items-center gap-2 shadow-lg backdrop-blur-md">
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
          <span>{stage}</span>
        </div>
      </div>
    </div>
  )
}
