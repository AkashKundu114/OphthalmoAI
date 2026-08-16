import { useState, useCallback, useEffect } from 'react'
import axios from 'axios'
import Cropper from 'react-easy-crop'
import getCroppedImg from './cropImage'
import {
  Upload, Activity, AlertTriangle, CheckCircle2,
  ChevronRight, Stethoscope, ShieldAlert, Pill,
  FileText, RefreshCw, Download, MapPin, Eye,
  ScanEye, Volume2, Layers, HelpCircle, ClipboardList,
  ShieldCheck, Microscope, Brain, Info, Github,
  ChevronDown, AlertCircle, VolumeX, Home,
  FlaskConical, GitBranch, BookOpen, Newspaper,
  ExternalLink, Search, Calendar, TrendingUp,
  ArrowRight, Sparkles, X, Send, Loader2, Bot, User,
  MessageCircle, Heart, Zap, Target, BarChart2,
  ChevronLeft, Star, Clock, Tag
} from 'lucide-react'
import jsPDF from 'jspdf'
import 'jspdf-autotable'
import ChatBot from './ChatBox'
const ACCENT = '#00ADB5'
const ACCENT_DARK = '#0891B2'
const NAVY = '#0F2040'

const FALLBACK_CONDITIONS = [
  { key: 'Cataract', name: 'Cataract', severity: 'Moderate to Severe', color: '#3B82F6', group: 'Anterior Segment' },
  { key: 'Uveitis', name: 'Uveitis', severity: 'High (Sight-Threatening)', color: '#EF4444', group: 'Anterior Segment' },
  { key: 'Conjunctivitis', name: 'Conjunctivitis', severity: 'Low (Contagious)', color: '#10B981', group: 'Ocular Surface' },
  { key: 'Jaundice', name: 'Jaundice', severity: 'High (Systemic Emergency)', color: '#F59E0B', group: 'Ocular Surface' },
  { key: 'Pterygium', name: 'Pterygium', severity: 'Moderate', color: '#8B5CF6', group: 'Ocular Surface' },
  { key: 'Eyelid', name: 'Eyelid Conditions', severity: 'Low', color: '#06B6D4', group: 'Adnexal/Oculoplastic' },
  { key: 'Normal', name: 'Normal Eye', severity: 'None', color: '#22C55E', group: 'All Groups' },
]

const urlToBase64 = (url) =>
  new Promise((resolve, reject) => {
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      canvas.getContext('2d').drawImage(img, 0, 0)
      resolve(canvas.toDataURL('image/jpeg', 0.85))
    }
    img.onerror = reject
    img.src = url
  })

const TabButton = ({ active, onClick, icon, label }) => (
  <button
    onClick={onClick}
    className={`px-4 py-2.5 rounded-xl flex items-center gap-2 text-xs font-semibold transition-all duration-200 ${
      active
        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-lg shadow-cyan-500/10'
        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60 border border-transparent'
    }`}
  >
    <span>{icon}</span>
    <span className="hidden sm:inline">{label}</span>
  </button>
)

const SymptomSelect = ({ label, value, setValue, options }) => (
  <div>
    <label className="block text-[11px] font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">
      {label}
    </label>
    <div className="relative">
      <select
        value={value}
        onChange={e => setValue(e.target.value)}
        className="w-full px-3 py-2 text-xs rounded-xl glass-input appearance-none text-slate-200 pr-8"
      >
        {options.map(opt => <option key={opt} value={opt} className="bg-slate-900 text-slate-200">{opt}</option>)}
      </select>
      <ChevronDown className="absolute w-3.5 h-3.5 -translate-y-1/2 pointer-events-none right-2.5 top-1/2 text-slate-400" />
    </div>
  </div>
)

const ProbabilityBar = ({ label, value }) => {
  const pct = Math.min(100, Math.max(0, value * 100))
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-300 font-medium">{label}</span>
        <span className="font-bold text-cyan-400 tabular-nums">{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full h-2 rounded-full bg-slate-800/80 overflow-hidden p-0.5 border border-slate-700/50">
        <div
          className="h-full rounded-full prob-bar-fill bg-gradient-to-r from-cyan-500 to-teal-400 shadow-sm"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

const SeverityBadge = ({ severity }) => {
  const s = (severity || '').toLowerCase()
  const isHigh = s.includes('high') || s.includes('severe') || s.includes('emergency')
  const isLow = s.includes('low') || s === 'none'
  return (
    <span className={`inline-flex items-center px-2.5 py-1 rounded-md text-[11px] font-semibold tracking-wide uppercase border ${
      isHigh ? 'bg-red-950/60 text-red-400 border-red-800/60'
      : isLow ? 'bg-emerald-950/60 text-emerald-400 border-emerald-800/60'
      : 'bg-amber-950/60 text-amber-400 border-amber-800/60'
    }`}>
      {severity}
    </span>
  )
}

const HomePage = ({ onNavigate }) => (
  <div className="space-y-12">
    {}
    <section className="relative overflow-hidden py-16 lg:py-24">
      <div className="max-w-6xl px-4 mx-auto sm:px-6 lg:px-8">
        <div className="grid items-center grid-cols-1 gap-12 lg:grid-cols-2">
          <div className="animate-fade-up space-y-6">
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold bg-cyan-950/80 text-cyan-300 border border-cyan-500/40 glow-teal">
              <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
              Next-Gen Ophthalmic AI Diagnostics
            </div>

            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-extrabold leading-tight tracking-tight">
              AI Retinal & Ocular <br />
              <span className="gradient-text">Disease Prediction</span>
            </h1>

            <p className="max-w-lg text-sm sm:text-base text-slate-300 leading-relaxed">
              Instant multi-disease ophthalmic screening across 7 clinical conditions using MobileNetV3 routing, EfficientNet-B4 specialist classification, and Grad-CAM visual heatmaps.
            </p>

            <div className="flex flex-wrap gap-4 pt-2">
              <button
                onClick={() => onNavigate('diagnostic')}
                className="inline-flex items-center gap-2.5 px-6 py-3.5 text-sm font-bold text-white rounded-xl bg-gradient-to-r from-cyan-500 via-teal-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 transition-all duration-300 shadow-xl shadow-cyan-500/25 hover:scale-105 active:scale-95"
              >
                <ScanEye className="w-5 h-5" /> Launch Screening Tool
              </button>

              <button
                onClick={() => onNavigate('workflow')}
                className="inline-flex items-center gap-2 px-5 py-3.5 text-sm font-medium text-slate-300 rounded-xl bg-slate-800/80 hover:bg-slate-700/80 hover:text-white border border-slate-700 transition-all duration-200"
              >
                How It Works <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            <div className="grid grid-cols-3 gap-4 pt-8 border-t border-slate-800">
              {[
                { value: '7 Conditions', label: 'Multi-Class AI Coverage' },
                { value: 'EfficientNet-B4', label: 'Deep Learning Engine' },
                { value: 'Grad-CAM', label: 'Explainable AI Heatmaps' },
              ].map((s, i) => (
                <div key={i}>
                  <p className="text-sm font-bold text-cyan-300">{s.value}</p>
                  <p className="text-[11px] text-slate-400 mt-0.5">{s.label}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-card p-6 rounded-3xl border border-slate-700/60 shadow-2xl relative">
            <p className="text-xs font-bold uppercase tracking-widest text-cyan-400 mb-4 flex items-center gap-2">
              <Brain className="w-4 h-4" /> Inference Pipeline Visualizer
            </p>

            <div className="space-y-3">
              {[
                { step: '01', label: 'Input Image Validation', desc: 'Magic bytes, MIME check & Quality Assessment', icon: <Upload className="w-4 h-4 text-cyan-400" /> },
                { step: '02', label: 'Anatomical Router', desc: 'MobileNetV3 routes to Adnexal, Anterior, or Surface', icon: <GitBranch className="w-4 h-4 text-indigo-400" /> },
                { step: '03', label: 'Specialist Prediction', desc: 'EfficientNet-B4 + Temperature Calibration + MC Dropout', icon: <Microscope className="w-4 h-4 text-teal-400" /> },
                { step: '04', label: 'Explainability & PDF', desc: 'Grad-CAM Heatmap + Clinical Code + PDF Exporter', icon: <FileText className="w-4 h-4 text-emerald-400" /> },
              ].map((item, i) => (
                <div key={i} className="flex items-center gap-3 p-3.5 bg-slate-900/90 rounded-xl border border-slate-800">
                  <div className="p-2.5 rounded-lg bg-slate-800/80 border border-slate-700">
                    {item.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-200">{item.label}</span>
                      <span className="text-[10px] font-mono text-cyan-400">{item.step}</span>
                    </div>
                    <p className="text-[11px] text-slate-400 truncate mt-0.5">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>

    {}
    <section className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { icon: <Zap className="w-6 h-6 text-amber-400" />, title: 'Real-Time Inference', desc: 'Sub-second GPU inference with calibrated probability confidence scoring.' },
          { icon: <ShieldCheck className="w-6 h-6 text-cyan-400" />, title: 'Hardened Security', desc: 'Strict input validation, rate limiting, and anonymized logging.' },
          { icon: <Bot className="w-6 h-6 text-emerald-400" />, title: 'Gemini Free Tier Chat', desc: 'Context-aware AI Doctor chat powered by Google Gemini API.' },
        ].map((f, i) => (
          <div key={i} className="glass-card p-6 rounded-2xl border border-slate-800 space-y-2">
            <div className="p-3 w-fit rounded-xl bg-slate-900 border border-slate-800">
              {f.icon}
            </div>
            <h3 className="text-base font-bold text-slate-100">{f.title}</h3>
            <p className="text-xs text-slate-400 leading-relaxed">{f.desc}</p>
          </div>
        ))}
      </div>
    </section>
  </div>
)

const MedicalNewsPage = () => {
  const [activeCategory, setActiveCategory] = useState('All')

  const categories = ['All', 'Research', 'Technology', 'Prevention', 'Treatment', 'Pediatric']

  const news = [
    {
      title: 'AI Outperforms Junior Doctors in Diagnosing Diabetic Retinopathy from Fundus Photos',
      category: 'Technology', date: 'May 2025', readTime: '5 min',
      summary: 'A multi-center study published in Nature Medicine demonstrated that a deep learning model achieved 94.5% sensitivity and 91.2% specificity in detecting referable diabetic retinopathy — outperforming three junior ophthalmologists under time-pressure conditions.',
      tags: ['AI', 'Diabetic Retinopathy', 'Deep Learning'],
      highlight: true,
      source: 'Nature Medicine',
    },
    {
      title: 'Omega-3 Fatty Acids Linked to 20% Reduction in Age-Related Macular Degeneration Risk',
      category: 'Prevention', date: 'April 2025', readTime: '4 min',
      summary: 'A 12-year longitudinal cohort study (n=38,000) found that participants with the highest dietary intake of DHA and EPA had a significantly lower incidence of early AMD compared to those with the lowest intake.',
      tags: ['AMD', 'Nutrition', 'Prevention'],
      highlight: false,
      source: 'JAMA Ophthalmology',
    },
    {
      title: 'Gene Therapy Trial Shows 70% Vision Improvement in Leber Congenital Amaurosis Patients',
      category: 'Treatment', date: 'April 2025', readTime: '6 min',
      summary: 'Phase III clinical trial results for AAV-mediated RPE65 gene therapy showed durable visual improvements at 3-year follow-up, with 70% of treated eyes meeting the primary endpoint of clinically meaningful functional vision improvement.',
      tags: ['Gene Therapy', 'LCA', 'Rare Disease'],
      highlight: false,
      source: 'The Lancet',
    },
    {
      title: 'Smartphone-Based OCT Reaches Primary Care — A Game Changer for Glaucoma Screening',
      category: 'Technology', date: 'March 2025', readTime: '4 min',
      summary: 'A portable, smartphone-attachable OCT device priced under $500 demonstrated 88% sensitivity for detecting glaucomatous optic nerve changes in a real-world primary care setting, opening the door to widespread community screening.',
      tags: ['Glaucoma', 'OCT', 'Telemedicine'],
      highlight: false,
      source: 'British Journal of Ophthalmology',
    },
    {
      title: 'Blue Light from Screens: Myth vs. Reality — A 2025 Meta-Analysis',
      category: 'Research', date: 'March 2025', readTime: '7 min',
      summary: 'A meta-analysis of 64 studies found no strong evidence linking screen-emitted blue light to permanent retinal damage in healthy adults. Evening screen use can still disrupt sleep timing.',
      tags: ['Blue Light', 'Digital Eye Strain', 'Meta-Analysis'],
      highlight: false,
      source: 'Investigative Ophthalmology & Visual Science',
    },
    {
      title: 'Global Myopia Crisis: 50% of World\'s Population Expected to Be Myopic by 2050',
      category: 'Research', date: 'February 2025', readTime: '5 min',
      summary: 'Updated modelling using global prevalence data projects that approximately 4.8 billion people will be myopic by 2050, with high myopia (≥−6D) affecting nearly 1 billion. Outdoor time and low-dose atropine remain the most evidence-backed interventions.',
      tags: ['Myopia', 'Epidemiology', 'Public Health'],
      highlight: true,
      source: 'The Lancet',
    },
    {
      title: 'Children\'s Vision After COVID-19: Screen Time and the Accelerated Myopia Surge',
      category: 'Pediatric', date: 'February 2025', readTime: '4 min',
      summary: 'Post-pandemic data from 12 Asian countries show a 2–3× acceleration in myopia progression among school-age children. The WHO recommends at least 2 hours of outdoor time daily as a primary preventative strategy.',
      tags: ['Myopia', 'Pediatric', 'COVID-19'],
      highlight: false,
      source: 'WHO Global Report',
    },
    {
      title: 'FDA Approves First Drop-Based Treatment for Presbyopia in Adults Over 40',
      category: 'Treatment', date: 'January 2025', readTime: '3 min',
      summary: 'A once-daily pilocarpine 1.25% ophthalmic solution received FDA approval for treating age-related near-vision blur. Clinical trials showed 30% of patients achieved ≥3 lines of improvement in corrected distance visual acuity.',
      tags: ['Presbyopia', 'FDA Approval', 'Treatment'],
      highlight: false,
      source: 'FDA Press Release',
    },
  ]

  const filtered = activeCategory === 'All' ? news : news.filter(n => n.category === activeCategory)

  const categoryColors = {
    Research: { bg: 'bg-indigo-950/60 text-indigo-400 border-indigo-900/60' },
    Technology: { bg: 'bg-blue-950/60 text-blue-400 border-blue-900/60' },
    Prevention: { bg: 'bg-emerald-950/60 text-emerald-400 border-emerald-900/60' },
    Treatment: { bg: 'bg-red-950/60 text-red-400 border-red-900/60' },
    Pediatric: { bg: 'bg-amber-950/60 text-amber-400 border-amber-900/60' },
  }

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Ophthalmic Research & Medical News</h2>
          <p className="text-xs text-slate-400 mt-1">Curated highlights and clinical findings from leading peer-reviewed journals and medical publications.</p>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setActiveCategory(cat)}
            className={`px-4 py-1.5 rounded-full text-xs font-semibold border transition-all duration-200 ${
              activeCategory === cat
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-lg shadow-cyan-500/10'
                : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            {cat}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {filtered.map((article, i) => {
          const catStyle = categoryColors[article.category] || { bg: 'bg-slate-950/60 text-slate-400 border-slate-900/60' }
          return (
            <div
              key={i}
              className={`glass-card p-6 rounded-2xl border transition-all flex flex-col justify-between ${
                article.highlight ? 'border-cyan-500/30 shadow-md shadow-cyan-500/5' : 'border-slate-800'
              }`}
            >
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-3">
                  <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded border ${catStyle.bg}`}>
                    {article.category}
                  </span>
                  <span className="flex items-center gap-1 text-[11px] text-slate-400">
                    <Calendar className="w-3.5 h-3.5" /> {article.date}
                  </span>
                  <span className="flex items-center gap-1 text-[11px] text-slate-400">
                    <Clock className="w-3.5 h-3.5" /> {article.readTime}
                  </span>
                </div>

                <div className="space-y-2">
                  {article.highlight && (
                    <div className="flex items-center gap-1 text-xs font-semibold text-cyan-400">
                      <Star className="w-3.5 h-3.5 fill-current" />
                      <span>Featured Publication</span>
                    </div>
                  )}
                  <h3 className="text-base font-bold text-slate-100 leading-snug">{article.title}</h3>
                  <p className="text-xs text-slate-400 leading-relaxed">{article.summary}</p>
                </div>
              </div>

              <div className="mt-4 pt-4 border-t border-slate-800/60 flex items-center justify-between">
                <span className="text-[10px] font-mono text-slate-500 italic">Source: {article.source}</span>
                <div className="flex flex-wrap gap-1">
                  {article.tags.map((tag, j) => (
                    <span key={j} className="text-[9px] px-2 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-400 font-mono">
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="p-4 rounded-xl bg-amber-950/20 border border-amber-900/30 flex items-start gap-3">
        <Info className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
        <p className="text-[11px] text-amber-200/80 leading-relaxed">
          <strong>Academic Advisory Note:</strong> These articles are summarized highlights for scientific education. They are not intended for clinical diagnosis. Always consult the original peer-reviewed source literature and certified practitioners for professional medical decision-making.
        </p>
      </div>
    </div>
  )
}

export default function App() {
  const [activeTab, setActiveTab] = useState('diagnostic')
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [cropping, setCropping] = useState(false)
  const [crop, setCrop] = useState({ x: 0, y: 0 })
  const [zoom, setZoom] = useState(1)
  const [croppedAreaPixels, setCroppedAreaPixels] = useState(null)

  
  const [painLevel, setPainLevel] = useState('None')
  const [visionLoss, setVisionLoss] = useState('No')
  const [itchiness, setItchiness] = useState('No')
  const [lightSensitivity, setLightSensitivity] = useState('No')
  const [floaters, setFloaters] = useState('No')
  const [discharge, setDischarge] = useState('None')
  const [duration, setDuration] = useState('Not Sure')
  const [halos, setHalos] = useState('No')

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showHeatmap, setShowHeatmap] = useState(true)

  const [conditions, setConditions] = useState(FALLBACK_CONDITIONS)
  const [searchQuery, setSearchQuery] = useState('')

  useEffect(() => {
    const fetchConditions = async () => {
      try {
        const apiUrl = import.meta.env.VITE_API_URL || '/api'
        const { data } = await axios.get(`${apiUrl}/conditions`)
        if (data && data.conditions) {
          setConditions(data.conditions)
        }
      } catch (e) {
        console.warn('Using fallback condition dataset:', e)
      }
    }
    fetchConditions()
  }, [])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedFile(file)
      setPreviewUrl(URL.createObjectURL(file))
      setCropping(true)
      setResult(null)
      setError(null)
    }
  }

  const onCropComplete = useCallback((_, croppedPixels) => {
    setCroppedAreaPixels(croppedPixels)
  }, [])

  const applyCrop = async () => {
    try {
      const croppedBlob = await getCroppedImg(previewUrl, croppedAreaPixels)
      setSelectedFile(croppedBlob)
      setPreviewUrl(URL.createObjectURL(croppedBlob))
      setCropping(false)
    } catch (e) {
      console.error('Crop error:', e)
      setCropping(false)
    }
  }

  const handleExportFHIR = async () => {
    const scanId = result?.scan_id || result?.id || 'DEMO-SCAN'
    try {
      const res = await axios.get(`${API_BASE_URL}/fhir/export/${scanId}`)
      const blob = new Blob([JSON.stringify(res.data, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `FHIR_Report_${scanId.slice(0, 8)}.json`
      a.click()
      URL.revokeObjectURL(url)
    } catch (e) {
      
      const fhirFallback = {
        resourceType: "DiagnosticReport",
        id: `ophthalmoai-${scanId}`,
        status: "final",
        code: { coding: [{ system: "http://snomed.info/sct", code: result?.snomed_code || "371405004", display: result?.diagnosis }] },
        conclusion: `AI Triage: ${result?.diagnosis} (${result?.confidence}% confidence). ICD-10: ${result?.icd10_code}`,
        issued: new Date().toISOString(),
      }
      const blob = new Blob([JSON.stringify(fhirFallback, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `FHIR_Report_${scanId.slice(0, 8)}.json`
      a.click()
      URL.revokeObjectURL(url)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const apiUrl = import.meta.env.VITE_API_URL || '/api'
      const formData = new FormData()
      formData.append('file', selectedFile, 'scan.jpg')
      formData.append('pain', painLevel)
      formData.append('vision', visionLoss)
      formData.append('itch', itchiness)
      formData.append('halos', halos)
      formData.append('discharge', discharge)
      formData.append('light_sens', lightSensitivity)
      formData.append('floaters', floaters)
      formData.append('duration', duration)

      const { data } = await axios.post(`${apiUrl}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setResult(data)
    } catch (err) {
      const detail = err?.response?.data?.detail || 'An unexpected error occurred during prediction analysis.'
      setError(typeof detail === 'string' ? detail : JSON.stringify(detail))
    } finally {
      setLoading(false)
    }
  }

  const generatePDFReport = async () => {
    if (!result) return
    const doc = new jsPDF()
    doc.setFillColor(15, 23, 42)
    doc.rect(0, 0, 210, 25, 'F')
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(16)
    doc.text('OphthalmoAI Clinical Screening Report', 14, 16)

    doc.setTextColor(30, 41, 59)
    doc.setFontSize(11)
    doc.text(`Primary Diagnosis: ${result.diagnosis}`, 14, 38)
    doc.text(`Confidence Score: ${result.confidence}%`, 14, 46)
    doc.text(`ICD-10 Code: ${result.icd10_code || 'N/A'}`, 14, 54)
    doc.text(`SNOMED-CT Code: ${result.snomed_code || 'N/A'}`, 14, 62)
    doc.text(`Urgency Classification: ${result.urgency || 'N/A'}`, 14, 70)

    doc.setFontSize(12)
    doc.text('Clinical Advice & Recommended Protocol:', 14, 85)
    doc.setFontSize(10)
    const adviceLines = doc.splitTextToSize(result.details?.advice || 'Consult an eye specialist.', 180)
    doc.text(adviceLines, 14, 93)

    if (result.hybrid_warnings && result.hybrid_warnings.length > 0) {
      doc.setFontSize(12)
      doc.text('Symptom Warnings & Cross-Check Alerts:', 14, 120)
      doc.setFontSize(9)
      let y = 128
      result.hybrid_warnings.forEach(w => {
        doc.text(`• ${w}`, 14, y)
        y += 6
      })
    }

    doc.save(`OphthalmoAI_Report_${result.diagnosis.replace(/\s+/g, '_')}.pdf`)
  }

  const filteredConditions = conditions.filter(c =>
    c.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    c.group.toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <div className="min-h-screen flex flex-col bg-[#090D16] text-slate-100 font-sans">
      {}
      <header className="sticky top-0 z-40 glass-panel border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3 cursor-pointer" onClick={() => setActiveTab('home')}>
              <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-teal-400 text-white shadow-lg shadow-cyan-500/20">
                <Eye className="w-5 h-5" />
              </div>
              <div>
                <span className="text-base font-extrabold tracking-wide text-white font-display">
                  Ophthalmo<span className="text-cyan-400">AI</span>
                </span>
                <span className="block text-[10px] text-slate-400 font-mono tracking-tight">
                  Clinical Retina Screening
                </span>
              </div>
            </div>

            {}
            <nav className="flex items-center gap-1 bg-slate-900/80 p-1.5 rounded-2xl border border-slate-800">
              <TabButton active={activeTab === 'home'} onClick={() => setActiveTab('home')} icon={<Home className="w-4 h-4" />} label="Home" />
              <TabButton active={activeTab === 'diagnostic'} onClick={() => setActiveTab('diagnostic')} icon={<ScanEye className="w-4 h-4" />} label="Diagnostic Tool" />
              <TabButton active={activeTab === 'conditions'} onClick={() => setActiveTab('conditions')} icon={<BookOpen className="w-4 h-4" />} label="Conditions" />
              <TabButton active={activeTab === 'workflow'} onClick={() => setActiveTab('workflow')} icon={<GitBranch className="w-4 h-4" />} label="Workflow" />
            </nav>

            <div className="hidden md:flex items-center gap-3">
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-800">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> AI System Online
              </span>
            </div>
          </div>
        </div>
      </header>

      {}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'home' && <HomePage onNavigate={setActiveTab} />}

        {activeTab === 'diagnostic' && (
          <div className="space-y-8 animate-fade-in">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-2xl font-bold text-white tracking-tight">AI Ocular Diagnostic Screening</h2>
                <p className="text-xs text-slate-400 mt-1">Upload a retinal scan, specify clinical symptoms, and obtain instant AI predictions.</p>
              </div>
              {result && (
                <button
                  onClick={generatePDFReport}
                  className="inline-flex items-center gap-2 px-4 py-2 text-xs font-bold rounded-xl bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 hover:bg-cyan-500/30 transition-all"
                >
                  <Download className="w-4 h-4" /> Download PDF Clinical Report
                </button>
              )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              {}
              <div className="lg:col-span-5 space-y-6">
                {}
                <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                  <p className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                    <Upload className="w-4 h-4" /> 1. Upload Eye Image
                  </p>

                  <div className="relative border-2 border-dashed border-slate-700/80 rounded-2xl p-6 text-center hover:border-cyan-500/60 transition-all duration-300 bg-slate-900/60 group">
                    <input
                      type="file"
                      accept="image/jpeg,image/png,image/bmp,image/webp"
                      onChange={handleFileChange}
                      className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                    />

                    {previewUrl ? (
                      <div className="relative space-y-3">
                        <img src={previewUrl} alt="Scan preview" className="max-h-48 mx-auto rounded-xl shadow-lg border border-slate-700 object-cover" />
                        <p className="text-[11px] text-cyan-400 font-medium">Click or drag to replace image</p>
                      </div>
                    ) : (
                      <div className="space-y-3 py-4">
                        <div className="w-12 h-12 rounded-full bg-cyan-500/10 text-cyan-400 flex items-center justify-center mx-auto border border-cyan-500/20 group-hover:scale-110 transition-transform">
                          <Upload className="w-6 h-6" />
                        </div>
                        <div>
                          <p className="text-xs font-semibold text-slate-200">Drag & drop eye scan or click to browse</p>
                          <p className="text-[10px] text-slate-400 mt-1">Supports JPEG, PNG, BMP, WEBP (Max 20MB)</p>
                        </div>
                      </div>
                    )}
                  </div>

                  {previewUrl && (
                    <button
                      onClick={() => setCropping(true)}
                      className="w-full py-2 text-xs font-semibold text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-xl border border-slate-700 transition-colors"
                    >
                      Crop & Focus Image Region
                    </button>
                  )}
                </div>

                {}
                <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                  <p className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-2">
                    <Stethoscope className="w-4 h-4" /> 2. Clinical Symptom Checklist
                  </p>

                  <div className="grid grid-cols-2 gap-3">
                    <SymptomSelect label="Eye Pain Level" value={painLevel} setValue={setPainLevel} options={['None', 'Mild', 'Severe']} />
                    <SymptomSelect label="Vision Loss" value={visionLoss} setValue={setVisionLoss} options={['No', 'Yes']} />
                    <SymptomSelect label="Itchiness" value={itchiness} setValue={setItchiness} options={['No', 'Yes']} />
                    <SymptomSelect label="Light Sensitivity" value={lightSensitivity} setValue={setLightSensitivity} options={['No', 'Yes']} />
                    <SymptomSelect label="Floaters" value={floaters} setValue={setFloaters} options={['No', 'Yes']} />
                    <SymptomSelect label="Discharge" value={discharge} setValue={setDischarge} options={['None', 'Watery', 'Thick/Yellow']} />
                    <SymptomSelect label="Duration" value={duration} setValue={setDuration} options={['Not Sure', '<1 week', '1-4 weeks', '>1 month']} />
                    <SymptomSelect label="Halos Around Light" value={halos} setValue={setHalos} options={['No', 'Yes']} />
                  </div>

                  <button
                    onClick={handleAnalyze}
                    disabled={!selectedFile || loading}
                    className="w-full py-3.5 text-xs font-bold text-white rounded-xl bg-gradient-to-r from-cyan-500 via-teal-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 transition-all duration-300 shadow-xl shadow-cyan-500/20 disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Running AI Diagnostic Models...
                      </>
                    ) : (
                      <>
                        <Activity className="w-4 h-4" />
                        Run Diagnostic Inference
                      </>
                    )}
                  </button>
                </div>
              </div>

              {}
              <div className="lg:col-span-7 space-y-6">
                {error && (
                  <div className="p-4 rounded-2xl bg-red-950/80 border border-red-800/80 text-red-200 text-xs flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                    <div>
                      <p className="font-bold">Diagnostic System Alert</p>
                      <p className="mt-0.5 leading-relaxed">{error}</p>
                    </div>
                  </div>
                )}

                {result ? (
                    <div className="space-y-6 animate-fade-up">
                      {}
                      <div className="glass-card p-6 rounded-3xl border border-emerald-500/20 shadow-2xl relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 via-cyan-500 to-teal-500"></div>
                        <div className="flex flex-col md:flex-row md:items-start justify-between gap-6 pb-6 border-b border-slate-800">
                          <div>
                            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[10px] font-bold uppercase tracking-widest bg-emerald-950/60 text-emerald-400 border border-emerald-800/60 mb-3">
                              {result.group_name}
                            </span>
                            <h3 className="text-3xl md:text-4xl font-extrabold text-white tracking-tight mb-2">{result.diagnosis}</h3>
                            <div className="flex flex-wrap gap-2 mt-2">
                              <span className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-[10px] text-slate-400 font-mono">ICD-10: <span className="text-emerald-300">{result.icd10_code || 'N/A'}</span></span>
                              <span className="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-[10px] text-slate-400 font-mono">SNOMED: <span className="text-emerald-300">{result.snomed_code || 'N/A'}</span></span>
                              <SeverityBadge severity={result.urgency || 'Normal'} />
                            </div>
                          </div>
                          <div className="text-right shrink-0">
                            <div className="text-4xl font-black text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400 tabular-nums">{result.confidence}%</div>
                            <span className="text-xs text-slate-400 font-mono font-medium block mt-1">Calibrated Confidence</span>
                            <span className="text-[10px] text-slate-500 font-mono block mt-0.5">MC Uncertainty: {(result.uncertainty * 100).toFixed(1)}%</span>
                          </div>
                        </div>

                        {}
                        {result.condition_details?.pathophysiology && (
                          <div className="pt-6">
                            <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2 mb-3">
                              <Brain className="w-4 h-4 text-emerald-400" /> Clinical Impression & Pathophysiology
                            </h4>
                            <p className="text-sm text-slate-400 leading-relaxed">
                              {result.condition_details.pathophysiology}
                            </p>
                          </div>
                        )}
                      </div>

                      {}
                      <div className="glass-panel p-6 rounded-2xl border border-slate-800">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-4 mb-4">
                          <h4 className="text-sm font-bold text-slate-200 flex items-center gap-2">
                            <Layers className="w-4 h-4 text-cyan-400" /> Visual Findings & Explainable AI
                          </h4>
                          <button onClick={() => setShowHeatmap(!showHeatmap)} className="text-[11px] px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-cyan-300 transition-colors">
                            {showHeatmap ? 'Toggle Original Scan' : 'Toggle Grad-CAM Heatmap'}
                          </button>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
                          <div className="md:col-span-2 space-y-3">
                            <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-900 group">
                              <img src={showHeatmap && result.heatmap ? result.heatmap : previewUrl} alt="Scan Analysis" className="w-full h-auto object-cover aspect-square transition-opacity duration-300" />
                              <div className="absolute top-2 right-2 px-2 py-1 rounded text-[9px] font-bold uppercase tracking-wider bg-black/60 text-white backdrop-blur-md">
                                {showHeatmap && result.heatmap ? 'Grad-CAM' : 'Original'}
                              </div>
                            </div>
                          </div>
                          <div className="md:col-span-3 space-y-4">
                            {result.condition_details?.analysis && (
                              <div>
                                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-1 block">Expected Visual Findings</span>
                                <p className="text-xs text-slate-300 leading-relaxed">{result.condition_details.analysis}</p>
                              </div>
                            )}
                            {result.spatial_description && (
                              <div className="bg-slate-950/60 p-3.5 border border-slate-850 rounded-xl space-y-1">
                                <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 block">AI Spatial Localization Annotation</span>
                                <p className="text-xs text-emerald-400 font-mono leading-relaxed">{result.spatial_description}</p>
                              </div>
                            )}
                            <div>
                              <span className="text-[10px] uppercase font-bold tracking-wider text-slate-500 mb-2 block">Specialist Model Probabilities</span>
                              <div className="space-y-2.5">
                                {Object.entries(result.probabilities || {}).map(([cls, prob]) => (
                                  <ProbabilityBar key={cls} label={cls} value={prob} />
                                ))}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {}
                        {result.condition_details?.diagnostic_workup && (
                          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                            <h4 className="text-sm font-bold text-indigo-400 flex items-center gap-2">
                              <Microscope className="w-4 h-4" /> Recommended Diagnostic Workup
                            </h4>
                            <ul className="space-y-2.5">
                              {result.condition_details.diagnostic_workup.map((workup, i) => (
                                <li key={i} className="flex items-start gap-2.5 text-xs text-slate-300 leading-relaxed">
                                  <span className="w-1.5 h-1.5 rounded-full bg-indigo-500/50 mt-1.5 shrink-0" />
                                  <span>{workup}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {}
                        {result.condition_details?.treatment && (
                          <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-4">
                            <h4 className="text-sm font-bold text-teal-400 flex items-center gap-2">
                              <Pill className="w-4 h-4" /> Clinical Treatment Protocols
                            </h4>
                            <ul className="space-y-2.5">
                              {result.condition_details.treatment.map((tx, i) => (
                                <li key={i} className="flex items-start gap-2.5 text-xs text-slate-300 leading-relaxed">
                                  <span className="w-1.5 h-1.5 rounded-full bg-teal-500/50 mt-1.5 shrink-0" />
                                  <span>{tx}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>

                      {}
                      <div className="glass-panel p-6 rounded-2xl border border-amber-900/30 bg-gradient-to-br from-slate-900 to-slate-950 space-y-5">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                          {result.condition_details?.precautions && (
                            <div className="space-y-3">
                              <h4 className="text-sm font-bold text-amber-400 flex items-center gap-2">
                                <ShieldAlert className="w-4 h-4" /> Immediate Precautions
                              </h4>
                              <ul className="space-y-2">
                                {result.condition_details.precautions.map((prec, i) => (
                                  <li key={i} className="flex items-start gap-2.5 text-xs text-amber-200/80 leading-relaxed">
                                    <AlertTriangle className="w-3.5 h-3.5 text-amber-500/70 shrink-0 mt-0.5" />
                                    <span>{prec}</span>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {result.condition_details?.doctor_notes && (
                            <div className="space-y-3">
                              <h4 className="text-sm font-bold text-cyan-400 flex items-center gap-2">
                                <Stethoscope className="w-4 h-4" /> Doctor's Clinical Notes
                              </h4>
                              <div className="p-4 rounded-xl bg-cyan-950/20 border border-cyan-900/30">
                                <p className="text-xs text-cyan-100/90 leading-relaxed italic">
                                  "{result.condition_details.doctor_notes}"
                                </p>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      {}
                      {result.hybrid_warnings_structured && result.hybrid_warnings_structured.length > 0 && (
                        <div className="glass-panel p-6 rounded-2xl border border-slate-800 space-y-3">
                          <p className="text-xs font-bold uppercase tracking-wider text-amber-400 flex items-center gap-2">
                            <ShieldAlert className="w-4 h-4" /> Clinical Symptom Alerts & Mismatches
                          </p>
                          <div className="space-y-2">
                            {result.hybrid_warnings_structured.map((w, idx) => (
                              <div
                                key={idx}
                                className={`p-3 rounded-xl border text-xs flex items-start gap-2.5 ${
                                  w.severity === 'urgent'
                                    ? 'bg-red-950/60 border-red-800/80 text-red-200'
                                    : w.severity === 'warning'
                                    ? 'bg-amber-950/60 border-amber-800/80 text-amber-200'
                                    : 'bg-cyan-950/60 border-cyan-800/80 text-cyan-200'
                                }`}
                              >
                                <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
                                <span className="leading-relaxed">{w.message}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {}
                      <div className="flex flex-col lg:flex-row gap-4">
                        {result.condition_details?.questions_for_doctor && (
                          <div className="flex-1 glass-panel p-5 rounded-2xl border border-slate-800 space-y-3">
                            <h4 className="text-xs font-bold text-slate-300 flex items-center gap-2 uppercase tracking-wider">
                              <ClipboardList className="w-4 h-4 text-emerald-400" /> Questions for your Ophthalmologist
                            </h4>
                            <ul className="space-y-2 pl-1">
                              {result.condition_details.questions_for_doctor.map((q, i) => (
                                <li key={i} className="flex items-start gap-2 text-[11px] text-slate-400">
                                  <span className="w-4 h-4 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 mt-0.5 text-[9px] text-emerald-400 font-mono">{i+1}</span>
                                  <span className="leading-relaxed mt-0.5">{q}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        <div className="lg:w-1/3 flex flex-col justify-end gap-3 p-5 glass-panel rounded-2xl border border-slate-800">
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider font-bold mb-1">Export Clinical Records</p>
                          <button
                            onClick={handleExportFHIR}
                            className="w-full px-4 py-2.5 rounded-xl text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-teal-300 border border-teal-800/60 transition flex items-center justify-center gap-2"
                          >
                            <Download className="w-4 h-4" />
                            <span>Export FHIR R4 Record</span>
                          </button>
                          <button
                            onClick={generatePDFReport}
                            className="w-full px-4 py-2.5 rounded-xl text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-500/20 transition flex items-center justify-center gap-2"
                          >
                            <FileText className="w-4 h-4" />
                            <span>Download Full PDF Report</span>
                          </button>
                        </div>
                      </div>
                    </div>
                ) : (
                  <div className="glass-panel p-12 rounded-2xl border border-slate-800 text-center space-y-4">
                    <div className="w-16 h-16 rounded-full bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-500">
                      <ScanEye className="w-8 h-8" />
                    </div>
                    <div className="max-w-xs mx-auto">
                      <p className="text-sm font-bold text-slate-300">No Active Screening Data</p>
                      <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                        Upload an eye scan on the left panel and click &quot;Run Diagnostic Inference&quot; to see prediction results and visual heatmaps.
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'conditions' && (
          <div className="space-y-6 animate-fade-in">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-2xl font-bold text-white">Clinical Conditions Directory</h2>
                <p className="text-xs text-slate-400 mt-1">Explore medical symptoms, clinical severities, and treatment protocols for detectable eye conditions.</p>
              </div>

              <div className="relative w-full sm:w-64">
                <input
                  type="text"
                  placeholder="Search conditions..."
                  value={searchQuery}
                  onChange={e => setSearchQuery(e.target.value)}
                  className="w-full px-3.5 py-2 pl-9 text-xs rounded-xl glass-input text-slate-200"
                />
                <Search className="absolute w-4 h-4 left-3 top-2.5 text-slate-400" />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredConditions.map(c => (
                <div key={c.key} className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded bg-slate-800 text-slate-300 border border-slate-700">
                      {c.group}
                    </span>
                    <SeverityBadge severity={c.severity} />
                  </div>

                  <h3 className="text-lg font-bold text-white">{c.name}</h3>
                  <p className="text-xs text-slate-400 leading-relaxed">{c.description || 'Clinical metadata provided by backend registry.'}</p>

                  {c.advice && (
                    <div className="p-3 bg-slate-900/90 rounded-xl border border-slate-800 text-xs text-cyan-300">
                      <span className="font-bold block text-[10px] uppercase text-cyan-400 mb-0.5">Clinical Protocol</span>
                      {c.advice}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'workflow' && (
          <div className="glass-panel p-8 rounded-3xl border border-slate-800 space-y-8 animate-fade-in">
            <div>
              <h2 className="text-2xl font-bold text-white">Deep Learning Inference Workflow</h2>
              <p className="text-xs text-slate-400 mt-1">Hierarchical multi-stage neural network architecture for high-accuracy ophthalmic screening.</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-2">
                <span className="text-xs font-mono font-bold text-cyan-400">Stage 1</span>
                <h4 className="text-sm font-bold text-white">Anatomical Group Router</h4>
                <p className="text-xs text-slate-400">MobileNetV3 classifies incoming scans into Adnexal Oculoplastic, Anterior Segment, or Ocular Surface.</p>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-2">
                <span className="text-xs font-mono font-bold text-teal-400">Stage 2</span>
                <h4 className="text-sm font-bold text-white">Specialist Classification</h4>
                <p className="text-xs text-slate-400">EfficientNet-B4 fine-tuned networks evaluate specific pathologies with temperature scaling calibration.</p>
              </div>

              <div className="p-5 rounded-2xl bg-slate-900/90 border border-slate-800 space-y-2">
                <span className="text-xs font-mono font-bold text-indigo-400">Stage 3</span>
                <h4 className="text-sm font-bold text-white">Symptom Cross-Check Engine</h4>
                <p className="text-xs text-slate-400">Rule-based expert system matches model outputs against reported patient symptoms to issue warnings.</p>
              </div>
            </div>
          </div>
        )}
      </main>

      {}
      {cropping && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md">
          <div className="glass-panel w-full max-w-lg p-6 rounded-3xl border border-slate-800 space-y-4">
            <h3 className="text-sm font-bold text-white">Crop & Adjust Eye Scan Region</h3>
            <div className="relative h-64 w-full bg-slate-950 rounded-2xl overflow-hidden border border-slate-800">
              <Cropper
                image={previewUrl}
                crop={crop}
                zoom={zoom}
                aspect={1}
                onCropChange={setCrop}
                onCropComplete={onCropComplete}
                onZoomChange={setZoom}
              />
            </div>
            <div className="flex items-center justify-end gap-3 pt-2">
              <button onClick={() => setCropping(false)} className="px-4 py-2 text-xs font-semibold text-slate-400 hover:text-white">Cancel</button>
              <button onClick={applyCrop} className="px-5 py-2 text-xs font-bold text-white bg-cyan-500 rounded-xl hover:bg-cyan-400 shadow-lg shadow-cyan-500/20">Apply Crop</button>
            </div>
          </div>
        </div>
      )}

      {}
      <ChatBot diagnosisContext={result ? { diagnosis: result.diagnosis, confidence: result.confidence, group_name: result.group_name, details: result.details } : null} />

      {}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-6">
        <div className="max-w-7xl mx-auto px-4 text-center text-xs text-slate-500">
          OphthalmoAI Retinal Disease Predictor · Clinical AI Screening Platform · Enterprise Security & Gemini Free Tier
        </div>
      </footer>
    </div>
  )
}
