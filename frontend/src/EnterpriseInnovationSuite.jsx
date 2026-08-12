import { useState, useEffect } from 'react'
import axios from 'axios'
import {
  Sparkles, Layers, ShieldCheck, Microscope, Zap, Target,
  TrendingUp, Calendar, RefreshCw, Send, CheckCircle2,
  AlertTriangle, Eye, Server, Radio, Database, Image as ImageIcon,
  Smartphone, User, MapPin, Clock, Info
} from 'lucide-react'

export default function EnterpriseInnovationSuite() {
  const apiUrl = import.meta.env.VITE_API_URL || '/api'
  
  
  const [subTab, setSubTab] = useState('triage') 

  
  const [pacsLog, setPacsLog] = useState([])
  const [pacsLoading, setPacsLoading] = useState(false)

  
  const [queue, setQueue] = useState([])
  const [queueLoading, setQueueLoading] = useState(false)
  const [signOffLoading, setSignOffLoading] = useState({})
  const [reviewDiagnosis, setReviewDiagnosis] = useState({})
  const [reviewNotes, setReviewNotes] = useState({})

  
  const [hba1c, setHba1c] = useState(6.2)
  const [systolic, setSystolic] = useState(130)
  const [diastolic, setDiastolic] = useState(85)
  const [age, setAge] = useState(55)
  const [isSmoker, setIsSmoker] = useState(false)
  const [modalDiagnosis, setModalDiagnosis] = useState('Cataract')
  const [modalConfidence, setModalConfidence] = useState(82)
  const [modalResult, setModalResult] = useState(null)
  const [modalLoading, setModalLoading] = useState(false)

  
  const [fedLogs, setFedLogs] = useState([])
  const [fedRound, setFedRound] = useState(12)
  const [fedAccuracy, setFedAccuracy] = useState(0.915)
  const [fedLoading, setFedLoading] = useState(false)

  
  const [compressFile, setCompressFile] = useState(null)
  const [compressResult, setCompressResult] = useState(null)
  const [compressLoading, setCompressLoading] = useState(false)
  const [compressPreview, setCompressPreview] = useState(null)

  
  const [syntheticCondition, setSyntheticCondition] = useState('Cataract')
  const [syntheticSeverity, setSyntheticSeverity] = useState('moderate')
  const [syntheticResult, setSyntheticResult] = useState(null)
  const [syntheticLoading, setSyntheticLoading] = useState(false)

  
  const [captureStatus, setCaptureStatus] = useState('idle') 
  const [blurScore, setBlurScore] = useState(0)
  const [centeringScore, setCenteringScore] = useState(0)
  const [captureMessage, setCaptureMessage] = useState('Click "Start Autocapture Simulator" to initiate smart edge loop.')

  
  const [clinicName, setClinicName] = useState('St. John Ophthalmic Clinic')
  const [appointmentTime, setAppointmentTime] = useState('')
  const [apptPurpose, setApptPurpose] = useState('Routine Cataract Check')
  const [apptHistory, setApptHistory] = useState([])
  const [apptLoading, setApptLoading] = useState(false)

  
  const fetchTriageQueue = async () => {
    setQueueLoading(true)
    try {
      const { data } = await axios.get(`${apiUrl}/admin/triage-queue`)
      setQueue(data.items || [])
    } catch (e) {
      console.error(e)
    } finally {
      setQueueLoading(false)
    }
  }

  const fetchAppointments = async () => {
    try {
      const { data } = await axios.get(`${apiUrl}/appointments/history`)
      setApptHistory(data.appointments || [])
    } catch (e) {
      console.error(e)
    }
  }

  useEffect(() => {
    fetchTriageQueue()
    fetchAppointments()
  }, [])

  
  const handlePacsImport = async () => {
    setPacsLoading(true)
    try {
      const { data } = await axios.post(`${apiUrl}/admin/pacs-import`)
      setPacsLog(prev => [
        {
          time: new Date().toLocaleTimeString(),
          msg: `PACS DICOM retrieved. Subject: ${data.dicom_metadata.dicom_patient_id}. Modality: ${data.dicom_metadata.dicom_modality}. Sync status: Success.`
        },
        ...prev
      ])
      fetchTriageQueue()
    } catch (e) {
      console.error(e)
    } finally {
      setPacsLoading(false)
    }
  }

  
  const handleSignOff = async (scanId) => {
    setSignOffLoading(prev => ({ ...prev, [scanId]: true }))
    try {
      const diag = reviewDiagnosis[scanId] || 'Normal'
      const note = reviewNotes[scanId] || ''
      const formData = new FormData()
      formData.append('verified_diagnosis', diag)
      formData.append('notes', note)
      
      const { data } = await axios.post(`${apiUrl}/scans/${scanId}/sign-off`, formData)
      
      setPacsLog(prev => [
        {
          time: new Date().toLocaleTimeString(),
          msg: `Scan ${scanId.substring(0,8)} signed off as ${diag}. EHR response: ${data.ehr_message}`
        },
        ...prev
      ])
      fetchTriageQueue()
    } catch (e) {
      console.error(e)
    } finally {
      setSignOffLoading(prev => ({ ...prev, [scanId]: false }))
    }
  }

  
  const handleMultiModal = async () => {
    setModalLoading(true)
    try {
      const { data } = await axios.post(`${apiUrl}/predict/multimodal`, {
        hba1c: parseFloat(hba1c),
        systolic_bp: parseInt(systolic),
        diastolic_bp: parseInt(diastolic),
        age: parseInt(age),
        is_smoker: isSmoker,
        diagnosis: modalDiagnosis,
        confidence: parseFloat(modalConfidence)
      })
      setModalResult(data)
    } catch (e) {
      console.error(e)
    } finally {
      setModalLoading(false)
    }
  }

  
  const handleFederatedSync = async () => {
    setFedLoading(true)
    try {
      const { data } = await axios.post(`${apiUrl}/admin/federated/sync`)
      setFedRound(data.global_round)
      setFedAccuracy(data.aggregated_accuracy)
      setFedLogs(data.node_updates || [])
    } catch (e) {
      console.error(e)
    } finally {
      setFedLoading(false)
    }
  }

  
  const handleCompressFile = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setCompressFile(file)
    setCompressPreview(URL.createObjectURL(file))
    
    setCompressLoading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const { data } = await axios.post(`${apiUrl}/scans/compress`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setCompressResult(data)
    } catch (err) {
      console.error(err)
    } finally {
      setCompressLoading(false)
    }
  }

  
  const handleGenerateSynthetic = async () => {
    setSyntheticLoading(true)
    try {
      const formData = new FormData()
      formData.append('severity', syntheticSeverity)
      const { data } = await axios.post(`${apiUrl}/admin/synthetic/generate?condition=${syntheticCondition}`, formData)
      setSyntheticResult(data)
    } catch (e) {
      console.error(e)
    } finally {
      setSyntheticLoading(false)
    }
  }

  
  const triggerAutocaptureSimulator = () => {
    setCaptureStatus('centering')
    setBlurScore(45)
    setCenteringScore(60)
    setCaptureMessage('Analysing stream: Centering pupil...')

    setTimeout(() => {
      setCenteringScore(95)
      setCaptureStatus('focusing')
      setCaptureMessage('Eye centered. Autofocus loop active...')
      
      setTimeout(() => {
        setBlurScore(145) 
        setCaptureStatus('measuring')
        setCaptureMessage('Perfect exposure. Capturing retina structure...')
        
        setTimeout(() => {
          setCaptureStatus('capturing')
          setCaptureMessage('Uploading edge-validated compressed payload...')
          
          setTimeout(() => {
            setCaptureStatus('done')
            setCaptureMessage('Image quality check passed. Autocapture completed successfully!')
          }, 1500)
        }, 1200)
      }, 1500)
    }, 1500)
  }

  
  const handleScheduleAppt = async () => {
    setApptLoading(true)
    try {
      await axios.post(`${apiUrl}/appointments`, {
        clinic_name: clinicName,
        appointment_time: appointmentTime || new Date(Date.now() + 86400000 * 2).toISOString(),
        purpose: apptPurpose
      })
      fetchAppointments()
      setApptPurpose('')
    } catch (e) {
      console.error(e)
    } finally {
      setApptLoading(false)
    }
  }

  return (
    <div className="space-y-8 animate-fade-in text-slate-100">
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-2xl font-bold text-white tracking-tight flex items-center gap-2">
          <Sparkles className="text-emerald-400 w-6 h-6 animate-pulse" /> Enterprise Innovation Workspace
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          A suite of 10 digital health pillars covering EHR integrations, Telemedicine triage queues, Federated learning nodes, and low-bandwidth simulations.
        </p>
      </div>

      {/* Sub tabs */}
      <div className="flex flex-wrap gap-2">
        {[
          { id: 'triage', label: 'EHR / PACS Triage Queue', icon: <Server className="w-3.5 h-3.5" /> },
          { id: 'multimodal', label: 'Multi-Modal Risk Lab', icon: <Microscope className="w-3.5 h-3.5" /> },
          { id: 'federated', label: 'Federated Node Aggregator', icon: <Radio className="w-3.5 h-3.5" /> },
          { id: 'assets', label: 'Synthetic & Compression Lab', icon: <Database className="w-3.5 h-3.5" /> },
          { id: 'client', label: 'Autocapture & CRM Hub', icon: <Smartphone className="w-3.5 h-3.5" /> },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setSubTab(tab.id)}
            className={`flex items-center gap-1.5 px-4 py-2 rounded-xl text-xs font-semibold border transition-all duration-200 ${
              subTab === tab.id
                ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-lg shadow-emerald-500/10'
                : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700'
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {}
      {subTab === 'triage' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {}
          <div className="lg:col-span-2 space-y-6">
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-base font-bold text-slate-200">Clinician Review Queue (Human-in-the-Loop)</h3>
                <span className="px-2.5 py-0.5 rounded text-[11px] font-mono bg-slate-950 border border-slate-800 text-slate-400">
                  {queue.length} Cases Pending
                </span>
              </div>

              {queueLoading ? (
                <div className="flex justify-center items-center py-12">
                  <RefreshCw className="w-8 h-8 text-emerald-400 animate-spin" />
                </div>
              ) : queue.length === 0 ? (
                <p className="text-xs text-slate-400 text-center py-12">No high-uncertainty scans require clinician sign-off at present.</p>
              ) : (
                <div className="space-y-4">
                  {queue.map((item) => (
                    <div key={item.scan_id} className="p-4 rounded-xl bg-slate-950/60 border border-slate-800 space-y-4">
                      <div className="flex justify-between items-start gap-4">
                        <div>
                          <p className="text-xs text-slate-400 font-mono">Case ID: {item.scan_id.substring(0,8)}...</p>
                          <h4 className="text-sm font-bold text-slate-200 mt-1">AI Classification: {item.diagnosis} ({item.confidence}%)</h4>
                          <p className="text-xs text-red-400 mt-1">Review Triggers: {item.reasons?.join('; ')}</p>
                        </div>
                        <div className="text-right">
                          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950/40 text-amber-400 border border-amber-800">
                            PENDING REVIEW
                          </span>
                        </div>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <label className="text-[11px] text-slate-400 block font-semibold">Verify or Override Diagnosis</label>
                          <select
                            value={reviewDiagnosis[item.scan_id] || item.diagnosis}
                            onChange={(e) => setReviewDiagnosis(prev => ({ ...prev, [item.scan_id]: e.target.value }))}
                            className="mt-1 w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 outline-none"
                          >
                            {['Normal', 'Cataract', 'Conjunctivitis', 'Eyelid', 'Pterygium', 'Uveitis', 'Jaundice'].map(d => (
                              <option key={d} value={d}>{d}</option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className="text-[11px] text-slate-400 block font-semibold">Clinician Case Notes</label>
                          <input
                            type="text"
                            placeholder="Add diagnostic comments..."
                            value={reviewNotes[item.scan_id] || ''}
                            onChange={(e) => setReviewNotes(prev => ({ ...prev, [item.scan_id]: e.target.value }))}
                            className="mt-1 w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-slate-200 outline-none"
                          />
                        </div>
                      </div>

                      <div className="flex justify-end">
                        <button
                          onClick={() => handleSignOff(item.scan_id)}
                          disabled={signOffLoading[item.scan_id]}
                          className="px-4 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition-all"
                        >
                          {signOffLoading[item.scan_id] ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <ShieldCheck className="w-3.5 h-3.5" />}
                          Sign-Off Case
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* PACS server simulation logs */}
          <div className="space-y-6">
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4 flex flex-col h-full">
              <h3 className="text-base font-bold text-slate-200">PACS / EHR Middleware</h3>
              
              <button
                onClick={handlePacsImport}
                disabled={pacsLoading}
                className="w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 transition-all"
              >
                {pacsLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Server className="w-4 h-4" />}
                Trigger PACS DICOM Import
              </button>

              <div className="flex-1 bg-slate-950 rounded-xl p-4 border border-slate-850 font-mono text-[11px] text-emerald-400 space-y-2 max-h-[300px] overflow-y-auto">
                <p className="text-slate-500 font-semibold">// Connection logs initialized.</p>
                {pacsLog.map((log, i) => (
                  <p key={i}>
                    <span className="text-slate-500">[{log.time}]</span> {log.msg}
                  </p>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: Multi-Modal Risk Lab */}
      {subTab === 'multimodal' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-slate-200">Systemic Biomarkers & Retinal Fusion</h3>
            <p className="text-xs text-slate-400">Enter patient physiological stats to fuse with standard retinal ML results.</p>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">Patient Age</label>
                <input
                  type="number"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-4 py-2 text-xs outline-none focus:border-slate-700 text-slate-200"
                />
              </div>
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">HbA1c Level (%)</label>
                <input
                  type="number"
                  step="0.1"
                  value={hba1c}
                  onChange={(e) => setHba1c(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-4 py-2 text-xs outline-none focus:border-slate-700 text-slate-200"
                />
              </div>
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">Systolic BP (mmHg)</label>
                <input
                  type="number"
                  value={systolic}
                  onChange={(e) => setSystolic(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-4 py-2 text-xs outline-none focus:border-slate-700 text-slate-200"
                />
              </div>
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">Diastolic BP (mmHg)</label>
                <input
                  type="number"
                  value={diastolic}
                  onChange={(e) => setDiastolic(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-4 py-2 text-xs outline-none focus:border-slate-700 text-slate-200"
                />
              </div>
            </div>

            <div className="flex items-center gap-6 py-2">
              <label className="flex items-center gap-2 text-xs text-slate-300 font-semibold cursor-pointer">
                <input
                  type="checkbox"
                  checked={isSmoker}
                  onChange={(e) => setIsSmoker(e.target.checked)}
                  className="rounded border-slate-800 bg-slate-900 text-emerald-500 focus:ring-0"
                />
                Active Smoker Status
              </label>
            </div>

            <div className="grid grid-cols-2 gap-4 border-t border-slate-850 pt-4">
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">Primary Diagnosis</label>
                <select
                  value={modalDiagnosis}
                  onChange={(e) => setModalDiagnosis(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-3 py-2 text-xs text-slate-200 outline-none"
                >
                  {['Normal', 'Cataract', 'Conjunctivitis', 'Eyelid', 'Pterygium', 'Uveitis', 'Jaundice'].map(d => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">ML Confidence (%)</label>
                <input
                  type="number"
                  value={modalConfidence}
                  onChange={(e) => setModalConfidence(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-4 py-2 text-xs outline-none focus:border-slate-700 text-slate-200"
                />
              </div>
            </div>

            <button
              onClick={handleMultiModal}
              disabled={modalLoading}
              className="w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 transition-all"
            >
              {modalLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Microscope className="w-4 h-4" />}
              Assess Multi-Modal Risk
            </button>
          </div>

          {}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 flex flex-col justify-between">
            <div>
              <h3 className="text-base font-bold text-slate-200 flex items-center gap-2">
                <Target className="text-emerald-400 w-5 h-5" /> Fused Multi-Modal Triage Report
              </h3>
              
              {!modalResult ? (
                <div className="flex flex-col items-center justify-center py-16 text-slate-500">
                  <Info className="w-12 h-12 stroke-[1.2] mb-2" />
                  <p className="text-xs">Submit biomarker details on the left to review prediction.</p>
                </div>
              ) : (
                <div className="space-y-6 mt-4">
                  {}
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-850 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-semibold text-slate-400">Cardiovascular Threat Level</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        modalResult.cardiovascular_risk_level === 'High' ? 'bg-red-950/40 text-red-400 border border-red-800' : 'bg-emerald-950/40 text-emerald-400 border border-emerald-800'
                      }`}>
                        {modalResult.cardiovascular_risk_level} (Score: {modalResult.cardiovascular_risk_score}/20)
                      </span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-1.5 border border-slate-800">
                      <div
                        className={`h-full rounded-full ${modalResult.cardiovascular_risk_level === 'High' ? 'bg-red-500' : 'bg-emerald-500'}`}
                        style={{ width: `${(modalResult.cardiovascular_risk_score / 20) * 100}%` }}
                      />
                    </div>
                  </div>

                  {}
                  <div className="p-4 rounded-xl bg-slate-950 border border-slate-850 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="text-xs font-semibold text-slate-400">Ocular Progression Risk</span>
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                        modalResult.ophthalmic_progression_risk_level === 'High' ? 'bg-red-950/40 text-red-400 border border-red-800' : 'bg-emerald-950/40 text-emerald-400 border border-emerald-800'
                      }`}>
                        {modalResult.ophthalmic_progression_risk_level} (Score: {modalResult.ophthalmic_progression_risk_score})
                      </span>
                    </div>
                    <div className="w-full bg-slate-900 rounded-full h-1.5 border border-slate-800">
                      <div
                        className={`h-full rounded-full ${modalResult.ophthalmic_progression_risk_level === 'High' ? 'bg-red-500' : 'bg-emerald-500'}`}
                        style={{ width: `${Math.min(100, (modalResult.ophthalmic_progression_risk_score / 2.0) * 100)}%` }}
                      />
                    </div>
                  </div>

                  {}
                  <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-850 flex gap-3">
                    <Info className="w-4 h-4 text-cyan-400 shrink-0 mt-0.5" />
                    <div>
                      <h4 className="text-xs font-bold text-slate-300">Clinical Decision Support Guidance</h4>
                      <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{modalResult.clinical_guidance}</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
            
            <p className="text-[10px] text-slate-500 mt-4 leading-relaxed italic">
              Multimodal fusion estimates vascular risks using the Framingham & retinopathy risk scoring models.
            </p>
          </div>
        </div>
      )}

      {}
      {subTab === 'federated' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {}
          <div className="lg:col-span-2 space-y-6">
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
              <div className="flex justify-between items-center">
                <h3 className="text-base font-bold text-slate-200">Local Research Node Sync status</h3>
                <button
                  onClick={handleFederatedSync}
                  disabled={fedLoading}
                  className="px-4 py-1.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition-all"
                >
                  {fedLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                  Trigger Global Weight Sync
                </button>
              </div>

              {fedLogs.length === 0 ? (
                <p className="text-xs text-slate-400 text-center py-12">No nodes aggregated. Click trigger above to start secure handshake.</p>
              ) : (
                <div className="space-y-4">
                  {fedLogs.map((node) => (
                    <div key={node.node_id} className="p-4 rounded-xl bg-slate-950/60 border border-slate-850 space-y-3">
                      <div className="flex justify-between items-center">
                        <div>
                          <h4 className="text-xs font-bold text-slate-200">{node.institution}</h4>
                          <p className="text-[10px] text-slate-500 font-mono">ID: {node.node_id}</p>
                        </div>
                        <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-950/50 text-emerald-400 border border-emerald-900/60">
                          SYNC SUCCESSFUL
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center text-xs font-mono border-t border-slate-900 pt-2 text-slate-400">
                        <div>
                          <span className="text-[10px] text-slate-500 block font-sans">Trained Samples</span>
                          {node.samples}
                        </div>
                        <div>
                          <span className="text-[10px] text-slate-500 block font-sans">Local Accuracy</span>
                          {(node.accuracy * 100).toFixed(2)}%
                        </div>
                        <div>
                          <span className="text-[10px] text-slate-500 block font-sans">Gradient Norm</span>
                          {node.gradients_norm}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {}
          <div className="space-y-6">
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-6">
              <h3 className="text-base font-bold text-slate-200">Global Aggregated Model</h3>
              
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-slate-950 border border-slate-850 text-center">
                  <span className="text-xs text-slate-500 block">Global Aggregation Round</span>
                  <span className="text-2xl font-bold text-white font-mono mt-1 block">{fedRound}</span>
                </div>

                <div className="p-4 rounded-xl bg-slate-950 border border-slate-850 text-center">
                  <span className="text-xs text-slate-500 block">Aggregated Global Accuracy</span>
                  <span className="text-2xl font-bold text-emerald-400 font-mono mt-1 block">{(fedAccuracy * 100).toFixed(2)}%</span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-950/40 border border-slate-850 flex gap-2">
                <Info className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <p className="text-[10px] text-slate-400 leading-relaxed">
                  <strong>Privacy Shield:</strong> The global aggregator aggregates model weights across hospitals securely using differential privacy without centralising patient data.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {}
      {subTab === 'assets' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-slate-200 flex items-center gap-2">
              <TrendingUp className="text-emerald-400 w-5 h-5" /> Low-Bandwidth Compression Protocol
            </h3>
            <p className="text-xs text-slate-400">
              Upload a test retinal photo to compress and evaluate bandwidth savings in remote clinics.
            </p>

            <div className="flex items-center justify-center w-full">
              <label className="flex flex-col items-center justify-center w-full h-32 border border-slate-800 border-dashed rounded-xl cursor-pointer hover:bg-slate-900/40 transition-all">
                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                  <ImageIcon className="w-8 h-8 text-slate-500 mb-2" />
                  <p className="text-xs text-slate-400">Select retinal scan for compression</p>
                </div>
                <input type="file" className="hidden" accept="image/*" onChange={handleCompressFile} />
              </label>
            </div>

            {compressLoading && (
              <div className="flex justify-center items-center py-4">
                <RefreshCw className="w-6 h-6 text-emerald-400 animate-spin" />
              </div>
            )}

            {compressResult && (
              <div className="p-4 rounded-xl bg-slate-950 border border-slate-850 space-y-4">
                <div className="grid grid-cols-2 gap-4 text-xs font-mono">
                  <div>
                    <span className="text-[10px] text-slate-500 font-sans block">Original Size</span>
                    {(compressResult.original_size_bytes / 1024).toFixed(1)} KB
                  </div>
                  <div>
                    <span className="text-[10px] text-slate-500 font-sans block">Compressed Size</span>
                    {(compressResult.compressed_size_bytes / 1024).toFixed(1)} KB
                  </div>
                </div>

                <div className="p-3 bg-emerald-950/20 border border-emerald-900/30 rounded-lg text-xs text-emerald-400 font-bold text-center">
                  Saved {compressResult.bandwidth_saved_percent}% Network Bandwidth!
                </div>
                
                {compressResult.compressed_file_base64 && (
                  <div className="border border-slate-800 rounded-lg overflow-hidden max-h-[180px] flex items-center justify-center bg-slate-900">
                    <img
                      src={`data:image/jpeg;base64,${compressResult.compressed_file_base64}`}
                      alt="Compressed Preview"
                      className="max-h-full max-w-full object-contain"
                    />
                  </div>
                )}
              </div>
            )}
          </div>

          {}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-slate-200">AI-Powered Synthetic Case Generator</h3>
            <p className="text-xs text-slate-400">Synthesize realistic fundus images to train clinicians/students.</p>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">Select Pathology</label>
                <select
                  value={syntheticCondition}
                  onChange={(e) => setSyntheticCondition(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-3 py-2 text-xs text-slate-200 outline-none"
                >
                  {['Cataract', 'Conjunctivitis', 'Eyelid', 'Pterygium', 'Uveitis', 'Jaundice'].map(d => (
                    <option key={d} value={d}>{d}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-[11px] text-slate-400 block font-semibold">Severity level</label>
                <select
                  value={syntheticSeverity}
                  onChange={(e) => setSyntheticSeverity(e.target.value)}
                  className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-xl px-3 py-2 text-xs text-slate-200 outline-none"
                >
                  <option value="mild">Mild</option>
                  <option value="moderate">Moderate</option>
                  <option value="severe">Severe</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleGenerateSynthetic}
              disabled={syntheticLoading}
              className="w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 transition-all"
            >
              {syntheticLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ImageIcon className="w-4 h-4" />}
              Generate Synthetic Fundus Case
            </button>

            {syntheticResult && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-1">
                    <span className="text-[10px] text-slate-500 font-semibold block text-center">Generated Fundus</span>
                    <div className="border border-slate-800 rounded-lg overflow-hidden h-[180px] bg-slate-950 flex items-center justify-center">
                      <img src={syntheticResult.synthetic_image_url} alt="Synthetic Fundus" className="max-h-full max-w-full object-contain" />
                    </div>
                  </div>
                  <div className="space-y-1">
                    <span className="text-[10px] text-slate-500 font-semibold block text-center">Grad-CAM Heatmap Target</span>
                    <div className="border border-slate-800 rounded-lg overflow-hidden h-[180px] bg-slate-950 flex items-center justify-center">
                      <img src={syntheticResult.synthetic_gradcam_url} alt="Synthetic Heatmap" className="max-h-full max-w-full object-contain animate-pulse" />
                    </div>
                  </div>
                </div>
                <p className="text-[10px] text-amber-500 italic text-center font-mono">{syntheticResult.disclaimer}</p>
              </div>
            )}
          </div>
        </div>
      )}

      {}
      {subTab === 'client' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
            <h3 className="text-base font-bold text-slate-200 flex items-center gap-2">
              <Smartphone className="text-emerald-400 w-5 h-5" /> Autocapture Edge IQA Simulator
            </h3>
            <p className="text-xs text-slate-400">Simulates real-time focus & alignment validations on edge devices.</p>

            <div className="relative border border-slate-800 bg-slate-950 rounded-xl h-48 flex flex-col justify-center items-center overflow-hidden">
              {}
              <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 opacity-20 pointer-events-none">
                {[...Array(9)].map((_, i) => (
                  <div key={i} className="border border-slate-700" />
                ))}
              </div>

              {captureStatus !== 'done' && captureStatus !== 'idle' && (
                <div className="relative w-20 h-20 rounded-full border border-dashed border-emerald-500/40 flex items-center justify-center animate-spin-slow">
                  <div className="w-14 h-14 rounded-full border border-emerald-500/60 flex items-center justify-center">
                    <Eye className="w-8 h-8 text-emerald-400 animate-pulse" />
                  </div>
                </div>
              )}

              {captureStatus === 'done' && (
                <CheckCircle2 className="w-12 h-12 text-emerald-400" />
              )}

              {captureStatus === 'idle' && (
                <Eye className="w-12 h-12 text-slate-600 stroke-[1.2]" />
              )}

              <p className="text-[11px] text-slate-400 font-mono mt-4 z-10">{captureMessage}</p>
            </div>

            <div className="grid grid-cols-2 gap-4 text-xs font-mono pt-2">
              <div className="p-3 bg-slate-950 border border-slate-850 rounded-xl text-center">
                <span className="text-[10px] text-slate-500 font-sans block">Autofocus Score</span>
                <span className={`${blurScore >= 80 ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}>
                  {blurScore} (Target: &ge;80)
                </span>
              </div>
              <div className="p-3 bg-slate-950 border border-slate-850 rounded-xl text-center">
                <span className="text-[10px] text-slate-500 font-sans block">Centering Alignment</span>
                <span className={`${centeringScore >= 90 ? 'text-emerald-400 font-bold' : 'text-slate-400'}`}>
                  {centeringScore}% (Target: &ge;90%)
                </span>
              </div>
            </div>

            <button
              onClick={triggerAutocaptureSimulator}
              disabled={captureStatus !== 'idle' && captureStatus !== 'done'}
              className="w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold py-2 rounded-xl text-xs flex items-center justify-center gap-1.5 transition-all"
            >
              Start Autocapture Simulator
            </button>
          </div>

          {}
          <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4 flex flex-col justify-between">
            <div>
              <h3 className="text-base font-bold text-slate-200 flex items-center gap-2">
                <Calendar className="text-emerald-400 w-5 h-5" /> Patient Navigation & CRM Reminders
              </h3>
              <p className="text-xs text-slate-400">Schedule follow-ups and simulate patient medical adherence SMS logs.</p>

              <div className="space-y-3 mt-3">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <div>
                    <label className="text-[10px] text-slate-400 font-semibold block">Select Ophthalmic Clinic</label>
                    <select
                      value={clinicName}
                      onChange={(e) => setClinicName(e.target.value)}
                      className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 outline-none"
                    >
                      <option value="St. John Ophthalmic Clinic">St. John Ophthalmic Clinic</option>
                      <option value="Moorfields Eye Outpost">Moorfields Eye Outpost</option>
                      <option value="Sankara Care Center">Sankara Care Center</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[10px] text-slate-400 font-semibold block">Booking Time</label>
                    <input
                      type="datetime-local"
                      value={appointmentTime}
                      onChange={(e) => setAppointmentTime(e.target.value)}
                      className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-lg px-2.5 py-1 text-xs text-slate-200 outline-none"
                    />
                  </div>
                </div>

                <div>
                  <label className="text-[10px] text-slate-400 font-semibold block">Appointment Purpose</label>
                  <input
                    type="text"
                    placeholder="e.g. Diabetic eye check..."
                    value={apptPurpose}
                    onChange={(e) => setApptPurpose(e.target.value)}
                    className="mt-1 w-full bg-slate-900 border border-slate-850 rounded-lg px-3 py-1.5 text-xs text-slate-200 outline-none"
                  />
                </div>

                <button
                  onClick={handleScheduleAppt}
                  disabled={apptLoading || !apptPurpose}
                  className="w-full bg-emerald-500 hover:bg-emerald-600 text-slate-950 font-bold py-1.5 rounded-lg text-xs flex items-center justify-center gap-1.5 transition-all"
                >
                  {apptLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Calendar className="w-3.5 h-3.5" />}
                  Schedule Follow-up
                </button>
              </div>
            </div>

            <div className="border-t border-slate-850 mt-4 pt-4 space-y-2">
              <h4 className="text-xs font-bold text-slate-300">Appointment Logs & SMS logs</h4>
              {apptHistory.length === 0 ? (
                <p className="text-[11px] text-slate-500 text-center py-4">No appointments scheduled.</p>
              ) : (
                <div className="space-y-2 max-h-[140px] overflow-y-auto">
                  {apptHistory.map((appt) => (
                    <div key={appt.id} className="p-2.5 rounded-lg bg-slate-950 border border-slate-850 text-[11px] space-y-1.5">
                      <div className="flex justify-between items-center text-slate-400 font-mono">
                        <span>{appt.clinic_name}</span>
                        <span className="text-emerald-400 font-bold uppercase text-[9px] bg-emerald-950/40 border border-emerald-900/60 px-1.5 rounded">
                          Scheduled
                        </span>
                      </div>
                      <p className="text-slate-200">Purpose: {appt.purpose}</p>
                      <p className="text-[10px] text-slate-500 flex items-center gap-1 font-mono">
                        <Clock className="w-3 h-3 text-cyan-400" /> {new Date(appt.time).toLocaleString()}
                      </p>
                      <p className="text-[9px] text-emerald-400 font-mono bg-emerald-950/20 border border-emerald-900/30 p-1.5 rounded mt-1">
                        📲 SMS SENT: "Hi! Your appointment at {appt.clinic_name} is set for {new Date(appt.time).toLocaleString()}. Keep eye health a priority."
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
