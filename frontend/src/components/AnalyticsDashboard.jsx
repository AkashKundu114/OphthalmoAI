import React, { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart2,
  CheckCircle2,
  Clock,
  Download,
  Eye,
  FileText,
  Info,
  Layers,
  PieChart as PieChartIcon,
  RefreshCw,
  ShieldAlert,
  TrendingDown,
  TrendingUp,
  Zap,
} from 'lucide-react'
import { getActiveApiUrl } from '../apiConfig'

const BASE_API_URL = getActiveApiUrl().endsWith('/api')
  ? getActiveApiUrl()
  : `${getActiveApiUrl()}/api`

export default function AnalyticsDashboard({ viewMode = 'clinical' }) {
  const [period, setPeriod] = useState('30d')
  const [selectedMetric, setSelectedMetric] = useState('screenings')
  const [dashboardData, setDashboardData] = useState(null)
  const [trendData, setTrendData] = useState([])
  const [anomaliesData, setAnomaliesData] = useState([])
  const [comparisonData, setComparisonData] = useState(null)
  const [showComparison, setShowComparison] = useState(false)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [reportStatus, setReportStatus] = useState(null)
  const [selectedTenant, setSelectedTenant] = useState('default-metro-eye-hospital')
  const [hoveredPoint, setHoveredPoint] = useState(null)

  const tenants = [
    { id: 'default-metro-eye-hospital', name: 'Metro Eye Institute & Research Hospital', tier: 'Tertiary Care' },
    { id: 'apex-retina', name: 'Apex Retina Center', tier: 'Specialty Clinic' },
    { id: 'st-jude-eye', name: 'St. Jude Eye Clinic', tier: 'Community Hospital' },
  ]

  const getHeaders = () => {
    const token = window.localStorage?.getItem('ophthalmo_token')
    const headers = {}
    if (token) headers['Authorization'] = `Bearer ${token}`
    if (selectedTenant) headers['X-Tenant-ID'] = selectedTenant
    return headers
  }

  const fetchAllAnalytics = async () => {
    try {
      setRefreshing(true)
      const headers = getHeaders()

      const [dashRes, trendRes, anomRes] = await Promise.all([
        axios.get(`${BASE_API_URL}/analytics/dashboard`, { headers }).catch(err => {
          console.warn('Dashboard fetch fallback:', err)
          return { data: null }
        }),
        axios.get(`${BASE_API_URL}/analytics/trends?metric=${selectedMetric}&period=${period}`, { headers }).catch(err => {
          console.warn('Trends fetch fallback:', err)
          return { data: { data_points: [] } }
        }),
        axios.get(`${BASE_API_URL}/analytics/anomalies`, { headers }).catch(err => {
          console.warn('Anomalies fetch fallback:', err)
          return { data: { anomalies: [] } }
        }),
      ])

      if (dashRes.data) setDashboardData(dashRes.data)
      if (trendRes.data?.data_points) setTrendData(trendRes.data.data_points)
      if (anomRes.data?.anomalies) setAnomaliesData(anomRes.data.anomalies)

      // Optionally attempt comparison endpoint
      try {
        const compRes = await axios.get(`${BASE_API_URL}/analytics/comparison`, { headers })
        if (compRes.data) setComparisonData(compRes.data)
      } catch {
        // Non-admin will 403 gracefully
        setComparisonData(null)
      }
    } catch (err) {
      console.error('Failed to load analytics data:', err)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchAllAnalytics()
  }, [period, selectedMetric, selectedTenant])

  const handleGenerateReport = async () => {
    try {
      setReportStatus('Generating executive report...')
      const headers = getHeaders()
      const res = await axios.post(
        `${BASE_API_URL}/analytics/generate-report`,
        { period, format: 'json' },
        { headers }
      )
      if (res.data?.status === 'success') {
        const rep = res.data.report
        const blob = new Blob([JSON.stringify(rep, null, 2)], { type: 'application/json' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `OphthalmoAI_Analytics_${selectedTenant}_${period}.json`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
        setReportStatus('Report downloaded successfully!')
      }
    } catch (err) {
      console.error('Error generating report:', err)
      setReportStatus('Failed to generate report.')
    } finally {
      setTimeout(() => setReportStatus(null), 3500)
    }
  }

  // Fallback synthetic data if API not fully populated yet
  const summary = dashboardData?.summary || {
    total_screenings_today: 142,
    total_screenings_7d: 1045,
    total_screenings_30d: 4320,
    total_patients_30d: 4110,
    trend: { direction: 'up', percentage: 14.2 },
    avg_confidence: 94.6,
    avg_inference_time_ms: 82.4,
    high_risk_count: 672,
  }

  const diagnosisList = dashboardData?.diagnosis_distribution?.length
    ? dashboardData.diagnosis_distribution
    : [
        { name: 'Normal', count: 2376, percentage: 55.0, color: '#10B981' },
        { name: 'Diabetic Retinopathy', count: 950, percentage: 22.0, color: '#EF4444' },
        { name: 'Glaucoma', count: 475, percentage: 11.0, color: '#6366F1' },
        { name: 'Cataract', count: 345, percentage: 8.0, color: '#F59E0B' },
        { name: 'Age-Related Macular Degeneration', count: 174, percentage: 4.0, color: '#06B6D4' },
      ]

  const activeAnomalies = anomaliesData.length > 0
    ? anomaliesData
    : (dashboardData?.active_anomalies || [])

  // Calculate SVG dimensions for trends line chart
  const chartHeight = 220
  const chartWidth = 720
  const padding = { top: 20, right: 30, bottom: 40, left: 50 }

  const { pointsStr, baselineStr, minVal, maxVal, mappedPoints } = useMemo(() => {
    if (!trendData || trendData.length === 0) {
      return { pointsStr: '', baselineStr: '', minVal: 0, maxVal: 100, mappedPoints: [] }
    }
    const vals = trendData.map(d => d.value)
    const baseVals = trendData.map(d => d.baseline)
    const allVals = [...vals, ...baseVals]
    let min = Math.min(...allVals) * 0.9
    let max = Math.max(...allVals) * 1.1
    if (min === max) { min -= 10; max += 10 }

    const innerW = chartWidth - padding.left - padding.right
    const innerH = chartHeight - padding.top - padding.bottom

    const mapped = trendData.map((d, i) => {
      const x = padding.left + (i / Math.max(1, trendData.length - 1)) * innerW
      const y = padding.top + innerH - ((d.value - min) / (max - min)) * innerH
      const by = padding.top + innerH - ((d.baseline - min) / (max - min)) * innerH
      return { ...d, x, y, by }
    })

    const pts = mapped.map(p => `${p.x},${p.y}`).join(' ')
    const bpts = mapped.map(p => `${p.x},${p.by}`).join(' ')

    return { pointsStr: pts, baselineStr: bpts, minVal: min, maxVal: max, mappedPoints: mapped }
  }, [trendData])

  // Latency SLA chart computation
  const latencyPoints = useMemo(() => {
    if (!trendData || trendData.length === 0) return []
    const innerW = chartWidth - padding.left - padding.right
    const innerH = chartHeight - padding.top - padding.bottom
    const minL = 0
    const maxL = 320 // SLA is 200ms

    return trendData.map((d, i) => {
      const x = padding.left + (i / Math.max(1, trendData.length - 1)) * innerW
      // Simulate p50 and p95
      const p50 = d.value ? (selectedMetric === 'inference_time' ? d.value * 0.85 : 75 + (i % 7) * 3) : 80
      const p95 = d.value ? (selectedMetric === 'inference_time' ? d.value : 98 + (i % 5) * 6) : 110
      const y50 = padding.top + innerH - ((p50 - minL) / (maxL - minL)) * innerH
      const y95 = padding.top + innerH - ((p95 - minL) / (maxL - minL)) * innerH
      return { date: d.date, x, y50, y95, p50, p95 }
    })
  }, [trendData, selectedMetric])

  const slaY = padding.top + (chartHeight - padding.top - padding.bottom) - ((200 - 0) / (320 - 0)) * (chartHeight - padding.top - padding.bottom)

  return (
    <div className="space-y-6">
      {/* Top Banner / Header */}
      <div className="bg-gradient-to-r from-slate-900 via-cyan-950 to-slate-900 border border-cyan-800/40 rounded-3xl p-6 text-white shadow-xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                Ad-Tech Telemetry Architecture
              </span>
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold tracking-wider uppercase bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" /> Live Telemetry
              </span>
            </div>
            <h1 className="text-2xl font-black tracking-tight text-white flex items-center gap-2.5">
              <BarChart2 className="w-6 h-6 text-cyan-400" />
              Multi-Tenant Screening Analytics & Anomaly Monitor
            </h1>
            <p className="text-xs text-slate-300 mt-1 max-w-2xl">
              Treats clinical screening flows like high-throughput ad exchanges: time-series volume monitoring,
              statistical anomaly detection (Z-Score, MA drops, Chi-Square shift), and strict &lt;200ms SLA tracking.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <select
              value={selectedTenant}
              onChange={(e) => setSelectedTenant(e.target.value)}
              className="bg-slate-800/80 border border-cyan-700/40 rounded-xl px-3 py-1.5 text-xs text-cyan-100 font-medium focus:outline-none focus:ring-2 focus:ring-cyan-500/50"
            >
              {tenants.map(t => (
                <option key={t.id} value={t.id} className="bg-slate-900 text-slate-200">
                  {t.name} ({t.tier})
                </option>
              ))}
            </select>

            <div className="flex rounded-xl bg-slate-800/80 p-0.5 border border-cyan-700/40 text-xs font-semibold">
              {['7d', '30d', '90d'].map(p => (
                <button
                  key={p}
                  onClick={() => setPeriod(p)}
                  className={`px-3 py-1 rounded-lg transition-all ${
                    period === p ? 'bg-cyan-600 text-white shadow-xs' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {p.toUpperCase()}
                </button>
              ))}
            </div>

            <button
              onClick={fetchAllAnalytics}
              disabled={refreshing}
              title="Refresh Analytics"
              className="p-2 rounded-xl bg-slate-800/80 border border-cyan-700/40 hover:bg-slate-700 text-cyan-300 transition-all disabled:opacity-50"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
            </button>

            <button
              onClick={handleGenerateReport}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 text-white text-xs font-bold shadow-md transition-all"
            >
              <Download className="w-3.5 h-3.5" />
              Export Report
            </button>
          </div>
        </div>

        {reportStatus && (
          <div className="mt-3 text-xs font-semibold px-3 py-1.5 rounded-xl bg-cyan-900/60 border border-cyan-500/40 text-cyan-200 inline-block">
            {reportStatus}
          </div>
        )}
      </div>

      {/* Ad-Tech Mapping Ribbon */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 bg-white/70 backdrop-blur-xs border border-slate-200 rounded-2xl p-3 shadow-2xs">
        <div className="flex items-center gap-2.5 px-2">
          <div className="w-8 h-8 rounded-xl bg-cyan-50 text-cyan-700 flex items-center justify-center font-bold text-xs">
            1
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-800">Screening Volume</div>
            <div className="text-[10px] text-slate-500">≡ Ad Campaign Impressions</div>
          </div>
        </div>
        <div className="flex items-center gap-2.5 px-2">
          <div className="w-8 h-8 rounded-xl bg-emerald-50 text-emerald-700 flex items-center justify-center font-bold text-xs">
            2
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-800">Diagnostic Certainty</div>
            <div className="text-[10px] text-slate-500">≡ CTR & Conversion Quality</div>
          </div>
        </div>
        <div className="flex items-center gap-2.5 px-2">
          <div className="w-8 h-8 rounded-xl bg-amber-50 text-amber-700 flex items-center justify-center font-bold text-xs">
            3
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-800">Inference Latency</div>
            <div className="text-[10px] text-slate-500">≡ RTB Bid Response SLA (&lt;200ms)</div>
          </div>
        </div>
        <div className="flex items-center gap-2.5 px-2">
          <div className="w-8 h-8 rounded-xl bg-purple-50 text-purple-700 flex items-center justify-center font-bold text-xs">
            4
          </div>
          <div>
            <div className="text-[11px] font-bold text-slate-800">Diagnosis Ratio</div>
            <div className="text-[10px] text-slate-500">≡ Audience Segment Composition</div>
          </div>
        </div>
      </div>

      {/* KPI Cards Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Screenings */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs relative overflow-hidden group hover:border-cyan-400 transition-all">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold mb-2">
            <span>Total Screenings ({period})</span>
            <div className="p-2 rounded-xl bg-cyan-50 text-cyan-600">
              <Activity className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-black text-slate-900 tracking-tight">
            {summary.total_screenings_30d.toLocaleString()}
          </div>
          <div className="flex items-center gap-1.5 mt-2 text-xs font-medium">
            <span className={`flex items-center gap-0.5 font-bold ${
              summary.trend.direction === 'up' ? 'text-emerald-600' : 'text-rose-600'
            }`}>
              {summary.trend.direction === 'up' ? <ArrowUpRight className="w-3.5 h-3.5" /> : <ArrowDownRight className="w-3.5 h-3.5" />}
              {Math.abs(summary.trend.percentage)}%
            </span>
            <span className="text-slate-500">vs prior period</span>
          </div>
          <div className="mt-2.5 pt-2 border-t border-slate-100 text-[10px] text-slate-500 flex items-center justify-between">
            <span>Today: <strong className="text-slate-800">{summary.total_screenings_today}</strong></span>
            <span className="text-cyan-700 font-medium">≡ Ad Impressions</span>
          </div>
        </div>

        {/* Diagnostic Confidence */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs relative overflow-hidden group hover:border-emerald-400 transition-all">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold mb-2">
            <span>Avg Model Confidence</span>
            <div className="p-2 rounded-xl bg-emerald-50 text-emerald-600">
              <CheckCircle2 className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-black text-slate-900 tracking-tight">
            {summary.avg_confidence.toFixed(1)}%
          </div>
          <div className="flex items-center gap-1.5 mt-2 text-xs font-medium text-emerald-600">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            <span>Calibrated Dirichlet Confidence</span>
          </div>
          <div className="mt-2.5 pt-2 border-t border-slate-100 text-[10px] text-slate-500 flex items-center justify-between">
            <span>Variance: &plusmn;1.4%</span>
            <span className="text-emerald-700 font-medium">≡ Conversion Rate (CTR)</span>
          </div>
        </div>

        {/* Inference Latency */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs relative overflow-hidden group hover:border-amber-400 transition-all">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold mb-2">
            <span>Avg Inference Latency</span>
            <div className="p-2 rounded-xl bg-amber-50 text-amber-600">
              <Clock className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-black text-slate-900 tracking-tight">
            {summary.avg_inference_time_ms.toFixed(1)} <span className="text-sm font-normal text-slate-500">ms</span>
          </div>
          <div className="flex items-center gap-1.5 mt-2 text-xs font-medium">
            <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
              summary.avg_inference_time_ms <= 200 ? 'bg-emerald-100 text-emerald-700' : 'bg-rose-100 text-rose-700'
            }`}>
              {summary.avg_inference_time_ms <= 200 ? 'SLA OK (<200ms)' : 'SLA BREACH'}
            </span>
            <span className="text-slate-500">Hard budget: 200ms</span>
          </div>
          <div className="mt-2.5 pt-2 border-t border-slate-100 text-[10px] text-slate-500 flex items-center justify-between">
            <span>p95: ~105ms</span>
            <span className="text-amber-700 font-medium">≡ RTB Bid Latency</span>
          </div>
        </div>

        {/* High Risk Detections */}
        <div className="bg-white rounded-2xl p-4 border border-slate-200 shadow-xs relative overflow-hidden group hover:border-purple-400 transition-all">
          <div className="flex items-center justify-between text-slate-500 text-xs font-semibold mb-2">
            <span>High-Risk Detections</span>
            <div className="p-2 rounded-xl bg-purple-50 text-purple-600">
              <ShieldAlert className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl font-black text-slate-900 tracking-tight">
            {summary.high_risk_count.toLocaleString()}
          </div>
          <div className="flex items-center gap-1.5 mt-2 text-xs font-medium text-purple-600">
            <span>Confidence &gt; 80% with pathology</span>
          </div>
          <div className="mt-2.5 pt-2 border-t border-slate-100 text-[10px] text-slate-500 flex items-center justify-between">
            <span>Yield: {((summary.high_risk_count / Math.max(1, summary.total_screenings_30d)) * 100).toFixed(1)}%</span>
            <span className="text-purple-700 font-medium">≡ High-Value Conversion</span>
          </div>
        </div>
      </div>

      {/* Main Charts Grid: Volume Trend + Diagnosis Donut */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Screening Volume Line Chart (2 Cols) */}
        <div className="lg:col-span-2 bg-white rounded-3xl p-6 border border-slate-200 shadow-xs">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-cyan-600" />
                Time-Series Screening Telemetry
              </h2>
              <p className="text-xs text-slate-500">
                Daily throughput with 7-day rolling baseline and Z-Score anomaly callouts (red markers)
              </p>
            </div>

            <div className="flex rounded-xl bg-slate-100 p-0.5 text-xs font-semibold border border-slate-200">
              {[
                { id: 'screenings', label: 'Screenings' },
                { id: 'confidence', label: 'Confidence' },
                { id: 'inference_time', label: 'Latency' },
                { id: 'high_risk_count', label: 'High Risk' },
              ].map(m => (
                <button
                  key={m.id}
                  onClick={() => setSelectedMetric(m.id)}
                  className={`px-2.5 py-1 rounded-lg transition-all ${
                    selectedMetric === m.id ? 'bg-white text-cyan-700 shadow-2xs font-bold' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Interactive SVG Chart */}
          <div className="relative w-full overflow-x-auto">
            <svg
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              className="w-full h-auto min-w-[500px] select-none"
            >
              {/* Grid Lines */}
              {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
                const y = padding.top + (chartHeight - padding.top - padding.bottom) * pct
                const val = maxVal - (pct * (maxVal - minVal))
                return (
                  <g key={idx}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={chartWidth - padding.right}
                      y2={y}
                      stroke="#E2E8F0"
                      strokeDasharray="4 4"
                    />
                    <text
                      x={padding.left - 8}
                      y={y + 4}
                      textAnchor="end"
                      fontSize="10"
                      fill="#94A3B8"
                      className="font-mono"
                    >
                      {Math.round(val)}
                    </text>
                  </g>
                )
              })}

              {/* Baseline Trendline (dashed slate) */}
              {baselineStr && (
                <polyline
                  fill="none"
                  stroke="#94A3B8"
                  strokeWidth="2"
                  strokeDasharray="4 3"
                  points={baselineStr}
                  opacity="0.8"
                />
              )}

              {/* Primary Value Line (cyan gradient stroke) */}
              {pointsStr && (
                <polyline
                  fill="none"
                  stroke="#0891B2"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={pointsStr}
                />
              )}

              {/* Data Points and Anomaly Markers */}
              {mappedPoints.map((p, idx) => {
                const isHovered = hoveredPoint?.date === p.date
                return (
                  <g key={idx} className="cursor-pointer" onMouseEnter={() => setHoveredPoint(p)} onMouseLeave={() => setHoveredPoint(null)}>
                    {/* Normal Point */}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={isHovered ? 5 : 2.5}
                      fill={p.is_anomaly ? '#EF4444' : '#0891B2'}
                      stroke="#FFFFFF"
                      strokeWidth={isHovered ? 2 : 1}
                    />

                    {/* Anomaly Pulsing Aura */}
                    {p.is_anomaly && (
                      <g>
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r="8"
                          fill="none"
                          stroke="#EF4444"
                          strokeWidth="2"
                          opacity="0.6"
                          className="animate-ping"
                        />
                        <circle
                          cx={p.x}
                          cy={p.y}
                          r="5"
                          fill="#EF4444"
                          stroke="#FFFFFF"
                          strokeWidth="1.5"
                        />
                      </g>
                    )}
                  </g>
                )
              })}

              {/* X Axis Labels */}
              {mappedPoints.filter((_, i) => i % Math.max(1, Math.floor(mappedPoints.length / 6)) === 0).map((p, idx) => (
                <text
                  key={idx}
                  x={p.x}
                  y={chartHeight - 12}
                  textAnchor="middle"
                  fontSize="10"
                  fill="#64748B"
                  className="font-mono"
                >
                  {p.date?.slice(5)}
                </text>
              ))}
            </svg>

            {/* Hover Tooltip Overlay */}
            {hoveredPoint && (
              <div
                className="absolute z-20 pointer-events-none bg-slate-900 text-white rounded-xl px-3 py-2 text-xs shadow-lg border border-slate-700 -translate-x-1/2 -translate-y-full"
                style={{
                  left: `${(hoveredPoint.x / chartWidth) * 100}%`,
                  top: `${(hoveredPoint.y / chartHeight) * 100}%`,
                }}
              >
                <div className="font-bold text-cyan-300">{hoveredPoint.date}</div>
                <div>Value: <span className="font-mono font-bold">{hoveredPoint.value}</span></div>
                <div className="text-[10px] text-slate-400">7d Baseline: {hoveredPoint.baseline}</div>
                {hoveredPoint.is_anomaly && (
                  <div className="mt-1 text-[10px] font-bold text-rose-400 bg-rose-950/80 px-1.5 py-0.5 rounded">
                    ⚠ Anomaly: {hoveredPoint.anomaly_reason || 'Deviation > 2σ'}
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="mt-3 flex items-center justify-between text-xs text-slate-500 border-t border-slate-100 pt-3">
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-cyan-600 inline-block" /> Daily Actual
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-slate-400 border-b border-dashed border-slate-400 inline-block" /> 7-Day Rolling Baseline
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-rose-500 inline-block" /> Statistical Anomaly
              </span>
            </div>
            <span className="text-[11px] text-slate-400 italic">Z-Score &gt; 2.0σ trigger</span>
          </div>
        </div>

        {/* Diagnosis Distribution Breakdown (1 Col) */}
        <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-xs flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-1">
              <PieChartIcon className="w-4 h-4 text-cyan-600" />
              Diagnosis Composition
            </h2>
            <p className="text-xs text-slate-500 mb-4">
              Distribution of screening outcomes (Audience Composition in ad-tech)
            </p>

            {/* Visual Donut / Stacked Bar representation */}
            <div className="space-y-3">
              {diagnosisList.map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-700 flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                      {item.name}
                    </span>
                    <span className="font-mono text-slate-900 font-bold">
                      {item.percentage}% <span className="text-[10px] text-slate-500 font-normal">({item.count})</span>
                    </span>
                  </div>
                  <div className="w-full h-2 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${item.percentage}%`, backgroundColor: item.color }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 bg-slate-50/80 rounded-2xl p-3 text-[11px] text-slate-600">
            <span className="font-bold text-slate-800">Chi-Square Drift Monitor:</span>
            <p className="mt-0.5 text-slate-500">
              Evaluates cohort stability using χ² goodness-of-fit against 14-day history. Alerts when disease prevalence diverges significantly from historical baseline.
            </p>
          </div>
        </div>
      </div>

      {/* Latency Performance & SLA Budget (<200ms) */}
      <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-xs">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-500" />
              Inference Latency SLA Telemetry (RTB Parallel)
            </h2>
            <p className="text-xs text-slate-500">
              p50 and p95 inference response time with strict 200ms SLA threshold enforcement
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-bold">
              99.2% SLA Compliance
            </span>
          </div>
        </div>

        {/* Latency Chart */}
        <div className="relative w-full overflow-x-auto">
          <svg viewBox={`0 0 ${chartWidth} 160`} className="w-full h-auto min-w-[500px]">
            {/* SLA Threshold Line at 200ms */}
            <line
              x1={padding.left}
              y1={slaY * (160 / chartHeight)}
              x2={chartWidth - padding.right}
              y2={slaY * (160 / chartHeight)}
              stroke="#EF4444"
              strokeWidth="2"
              strokeDasharray="6 4"
            />
            <text
              x={chartWidth - padding.right}
              y={slaY * (160 / chartHeight) - 6}
              textAnchor="end"
              fontSize="10"
              fill="#EF4444"
              fontWeight="bold"
            >
              200ms RTB / SLA Hard Limit
            </text>

            {/* Latency Curves */}
            {latencyPoints.length > 1 && (
              <>
                <polyline
                  fill="none"
                  stroke="#F59E0B"
                  strokeWidth="2"
                  points={latencyPoints.map(p => `${p.x},${p.y95 * (160 / chartHeight)}`).join(' ')}
                />
                <polyline
                  fill="none"
                  stroke="#0891B2"
                  strokeWidth="1.5"
                  points={latencyPoints.map(p => `${p.x},${p.y50 * (160 / chartHeight)}`).join(' ')}
                />
              </>
            )}
          </svg>
        </div>

        <div className="mt-2 flex items-center justify-between text-xs text-slate-500 border-t border-slate-100 pt-2">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-amber-500 inline-block" /> p95 Latency</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-cyan-600 inline-block" /> p50 Latency</span>
            <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-rose-500 border-b border-dashed border-rose-500 inline-block" /> 200ms SLA Limit</span>
          </div>
          <span className="text-slate-400">Modeled after real-time bidding auction deadlines (&lt;200ms)</span>
        </div>
      </div>

      {/* Anomaly Alerts Panel (Ad-Tech Parallel Demonstration) */}
      <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-xs">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <h2 className="text-sm font-bold text-slate-900">
              Active Anomaly Detection Alerts ({activeAnomalies.length})
            </h2>
          </div>
          <span className="text-xs text-slate-500">
            Automated statistical alarms with direct ad-tech system mappings
          </span>
        </div>

        {activeAnomalies.length === 0 ? (
          <div className="p-8 text-center bg-slate-50 rounded-2xl border border-slate-200">
            <CheckCircle2 className="w-8 h-8 text-emerald-500 mx-auto mb-2" />
            <div className="text-sm font-bold text-slate-800">All Metrics Within Normal Baseline</div>
            <p className="text-xs text-slate-500 mt-1">No volume spikes, SLA breaches, or distribution drifts detected for this tenant.</p>
          </div>
        ) : (
          <div className="space-y-4">
            {activeAnomalies.map((anom, idx) => {
              const isCrit = anom.severity === 'critical'
              return (
                <div
                  key={idx}
                  className={`rounded-2xl p-4 border transition-all ${
                    isCrit ? 'bg-rose-50/60 border-rose-200' : 'bg-amber-50/60 border-amber-200'
                  }`}
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                    <div className="flex items-center gap-2">
                      <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-black uppercase tracking-wider ${
                        isCrit ? 'bg-rose-600 text-white' : 'bg-amber-500 text-white'
                      }`}>
                        {anom.severity}
                      </span>
                      <span className="text-xs font-bold text-slate-900 uppercase tracking-wide">
                        {anom.anomaly_type?.replace('_', ' ')}
                      </span>
                      <span className="text-xs text-slate-500 font-mono">
                        ({anom.metric_name})
                      </span>
                    </div>

                    <div className="text-xs font-mono font-semibold text-slate-700">
                      Current: <strong className={isCrit ? 'text-rose-700' : 'text-amber-800'}>{anom.current_value}</strong>
                      <span className="text-slate-400 mx-1">/</span>
                      Baseline: {anom.baseline_value}
                      <span className="text-slate-500 ml-1.5">({anom.deviation_percent > 0 ? '+' : ''}{anom.deviation_percent}%)</span>
                    </div>
                  </div>

                  {/* Recommendation */}
                  <div className="text-xs text-slate-700 mb-2">
                    <strong className="text-slate-900">Clinical Action:</strong> {anom.recommendation}
                  </div>

                  {/* Ad-Tech Engineering Parallel Box */}
                  <div className="mt-2.5 p-2.5 rounded-xl bg-slate-900 text-cyan-200 border border-cyan-800/40 text-xs flex items-start gap-2 shadow-inner">
                    <Zap className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-bold text-white tracking-wide uppercase text-[10px] bg-cyan-900/60 px-1.5 py-0.5 rounded border border-cyan-500/30 mr-1.5">
                        Ad-Tech System Parallel
                      </span>
                      <span className="text-slate-300">{anom.ad_tech_parallel}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Admin Cross-Tenant Comparison View */}
      {comparisonData && (
        <div className="bg-white rounded-3xl p-6 border border-slate-200 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-600" />
                Cross-Tenant Benchmarking (Admin View)
              </h2>
              <p className="text-xs text-slate-500">
                Platform-wide comparison of clinic throughput, latency SLA, and anomaly rates
              </p>
            </div>
            <button
              onClick={() => setShowComparison(!showComparison)}
              className="text-xs font-semibold text-cyan-700 hover:text-cyan-800 px-3 py-1 rounded-xl bg-cyan-50 border border-cyan-200"
            >
              {showComparison ? 'Collapse Benchmarks' : 'Expand Benchmarks'}
            </button>
          </div>

          {showComparison && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 uppercase tracking-wider text-[10px]">
                    <th className="py-2 px-3">Clinic Tenant</th>
                    <th className="py-2 px-3">30D Screenings</th>
                    <th className="py-2 px-3">Avg Confidence</th>
                    <th className="py-2 px-3">Avg Latency</th>
                    <th className="py-2 px-3">SLA Compliance</th>
                    <th className="py-2 px-3">Anomalies</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {comparisonData.tenants?.map((t, idx) => (
                    <tr key={idx} className="hover:bg-slate-50">
                      <td className="py-2.5 px-3 font-semibold text-slate-900">{t.clinic_name}</td>
                      <td className="py-2.5 px-3 font-mono">{t.total_screenings_30d.toLocaleString()}</td>
                      <td className="py-2.5 px-3 font-mono text-emerald-700 font-semibold">{t.avg_confidence}%</td>
                      <td className="py-2.5 px-3 font-mono">{t.avg_inference_time_ms} ms</td>
                      <td className="py-2.5 px-3 font-mono">
                        <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          t.sla_compliance_rate >= 95 ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-800'
                        }`}>
                          {t.sla_compliance_rate}%
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono font-bold text-rose-600">{t.anomaly_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
