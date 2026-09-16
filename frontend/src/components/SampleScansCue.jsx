import React from 'react'
import { Sparkles, CheckCircle2, XCircle, Info, ScanEye } from 'lucide-react'
import { playClickSound } from '../utils/soundEffects'

const FUNDUS_SAMPLES = [
  {
    id: 'normal',
    label: 'Normal Fundus',
    category: 'Healthy Retina',
    badgeColor: 'text-emerald-700 bg-emerald-50 border-emerald-200',
    path: '/samples/fundus_normal.jpg',
    desc: 'Sharp neuroretinal rim, uniform retinal perfusion, distinct foveal reflex.'
  },
  {
    id: 'diabetic_retinopathy',
    label: 'Diabetic Retinopathy',
    category: 'Microvascular',
    badgeColor: 'text-rose-700 bg-rose-50 border-rose-200',
    path: '/samples/fundus_diabetic_retinopathy.jpg',
    desc: 'Microaneurysms, punctate intraretinal hemorrhages, hard lipid exudates.'
  },
  {
    id: 'glaucoma',
    label: 'Glaucoma',
    category: 'Optic Neuropathy',
    badgeColor: 'text-purple-700 bg-purple-50 border-purple-200',
    path: '/samples/fundus_glaucoma.jpg',
    desc: 'Increased cup-to-disc ratio (CDR > 0.7) and neuroretinal rim thinning.'
  },
  {
    id: 'amd',
    label: 'Macular Degeneration',
    category: 'Maculopathy',
    badgeColor: 'text-amber-700 bg-amber-50 border-amber-200',
    path: '/samples/fundus_amd.jpg',
    desc: 'Central confluent drusen deposits and retinal pigment epithelial atrophy.'
  },
  {
    id: 'cataract',
    label: 'Cataract Media Haze',
    category: 'Optical Media',
    badgeColor: 'text-cyan-700 bg-cyan-50 border-cyan-200',
    path: '/samples/fundus_cataract.jpg',
    desc: 'Nuclear lens opacity causing diffuse contrast attenuation and illumination scatter.'
  },
  {
    id: 'myopia',
    label: 'Hypertensive / Myopia',
    category: 'Vascular / Axial',
    badgeColor: 'text-blue-700 bg-blue-50 border-blue-200',
    path: '/samples/fundus_myopia.jpg',
    desc: 'Arteriolar crossing changes, peripapillary atrophy, and myopic conus.'
  },
]

/**
 * SampleScansCue - Interactive cue cards and sample fundus photograph loader inspired by cuedesign.space & shotbase.com
 */
export default function SampleScansCue({ onSelectSample }) {
  const handleSelect = async (sample) => {
    playClickSound()
    try {
      const res = await fetch(sample.path)
      const blob = await res.blob()
      const file = new File([blob], `${sample.id}_fundus.jpg`, { type: 'image/jpeg' })
      onSelectSample(file, sample.path)
    } catch {
      onSelectSample(null, sample.path)
    }
  }

  return (
    <div className="space-y-4">
      {/* Quick Test Fundus Samples */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold uppercase tracking-wider text-slate-700 flex items-center gap-1.5">
            <ScanEye className="w-4 h-4 text-cyan-600" />
            Clinical Fundus Samples
          </span>
          <span className="text-[11px] text-slate-500 font-medium">Select a scan to test the neural ensemble</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5">
          {FUNDUS_SAMPLES.map((sample) => (
            <button
              key={sample.id}
              onClick={() => handleSelect(sample)}
              className="group relative p-2 rounded-xl bg-white border border-slate-200 hover:border-cyan-500 hover:shadow-md transition-all text-left flex flex-col justify-between overflow-hidden cursor-pointer"
            >
              <div className="relative aspect-square w-full rounded-lg overflow-hidden mb-1.5 bg-slate-950 border border-slate-800 flex items-center justify-center">
                <img
                  src={sample.path}
                  alt={sample.label}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  loading="lazy"
                />
                <span className={`absolute bottom-1 left-1 px-1.5 py-0.5 rounded font-mono text-[9px] font-bold border backdrop-blur-xs ${sample.badgeColor}`}>
                  {sample.category}
                </span>
              </div>
              <div>
                <h4 className="text-[11px] font-bold text-slate-900 group-hover:text-cyan-700 transition line-clamp-1">
                  {sample.label}
                </h4>
                <p className="text-[10px] text-slate-500 line-clamp-2 leading-tight mt-0.5">
                  {sample.desc}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Visual Quality Cue Cards (cuedesign.space) */}
      <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-xs">
        <div className="flex items-center gap-1.5 font-bold text-slate-800 mb-2">
          <Info className="w-3.5 h-3.5 text-cyan-600" />
          <span>Retinal Fundus Photography Quality Criteria</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px]">
          <div className="flex items-start gap-2 bg-white p-2 rounded-lg border border-emerald-200">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold text-emerald-800 block">High Quality Fundus Scan</span>
              <span className="text-slate-600">Centered optic disc and macula, sharp vascular tree bifurcation, uniform illumination with full circular aperture.</span>
            </div>
          </div>
          <div className="flex items-start gap-2 bg-white p-2 rounded-lg border border-rose-200">
            <XCircle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold text-rose-800 block">Common Capture Pitfalls</span>
              <span className="text-slate-600">Corneal glare flash reflection, pupil edge shadow vignetting, eyelid/lash obscuration, or severe motion defocus.</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
