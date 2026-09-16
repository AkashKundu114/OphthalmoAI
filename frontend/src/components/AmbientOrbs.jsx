import React from 'react'

/**
 * AmbientOrbs - Subtle ambient chromatic floating orbs inspired by libraries.dev/orbs
 * Rendered behind content with pointer-events-none and heavy gaussian blur for a pristine clinical glow.
 */
export default function AmbientOrbs() {
  return (
    <div className="fixed inset-0 pointer-events-none -z-10 overflow-hidden" aria-hidden="true">
      {/* Cyan primary glow orb */}
      <div 
        className="absolute -top-32 left-1/4 w-[540px] h-[540px] rounded-full bg-cyan-400/10 blur-[130px] animate-pulse" 
        style={{ animationDuration: '8s' }}
      />
      {/* Violet/Indigo secondary depth orb */}
      <div 
        className="absolute top-1/3 -right-32 w-[480px] h-[480px] rounded-full bg-indigo-500/[0.07] blur-[140px] animate-pulse" 
        style={{ animationDuration: '12s', animationDelay: '2s' }}
      />
      {/* Emerald health accent orb */}
      <div 
        className="absolute bottom-1/4 -left-20 w-[420px] h-[420px] rounded-full bg-emerald-400/[0.08] blur-[120px] animate-pulse" 
        style={{ animationDuration: '10s', animationDelay: '4s' }}
      />
      {/* Subtle fine geometric grid pattern overlay inspired by bookofshapes.com */}
      <div 
        className="absolute inset-0 opacity-[0.035]"
        style={{
          backgroundImage: `linear-gradient(to right, #0F172A 1px, transparent 1px), linear-gradient(to bottom, #0F172A 1px, transparent 1px)`,
          backgroundSize: '36px 36px',
        }}
      />
    </div>
  )
}
