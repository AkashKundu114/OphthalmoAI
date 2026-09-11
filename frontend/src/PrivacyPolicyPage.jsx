import React, { useState } from 'react'
import {
  ShieldCheck, Lock, Eye, ArrowLeft, Database,
  FileCheck, Server, AlertCircle, CheckCircle2,
  Cpu, FileText, Mail, Calendar, HelpCircle,
  ExternalLink, Key, RefreshCw, UserCheck
} from 'lucide-react'

export default function PrivacyPolicyPage({ onNavigate }) {
  const [activeTab, setActiveTab] = useState('all')

  const policySections = [
    {
      id: 'commitment',
      icon: <ShieldCheck className="w-5 h-5 text-teal-400" />,
      badge: 'Core Ethics',
      badgeColor: 'bg-teal-950/80 text-teal-300 border-teal-800',
      title: '1. Privacy Philosophy & Clinical Data Ethics',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            At <strong>OphthalmoAI</strong>, we operate with an uncompromising commitment to patient confidentiality, biomedical data ethics, and regulatory transparency. Ophthalmic imagery captures vital windows into human health, requiring privacy safeguards that meet or exceed international clinical standards.
          </p>
          <div className="p-4 rounded-xl bg-teal-950/30 border border-teal-800/60 text-teal-200">
            <span className="font-bold text-teal-300 block mb-1">Zero Commercial Monetization Guarantee</span>
            We do <strong>not</strong> sell, rent, broker, or monetize patient retinal scans, ophthalmic photographs, symptom histories, or clinical inferences under any circumstances.
          </div>
        </div>
      )
    },
    {
      id: 'data-collected',
      icon: <Database className="w-5 h-5 text-cyan-400" />,
      badge: 'Data Inventory',
      badgeColor: 'bg-cyan-950/80 text-cyan-300 border-cyan-800',
      title: '2. Clinical & Operational Data We Process',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            When utilizing OphthalmoAI for ocular triage and screening, we process only the specific data fields required to compute neural network inferences and generate clinical reports:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
              <span className="font-bold text-white text-xs flex items-center gap-2">
                <Eye className="w-4 h-4 text-cyan-400" /> Ocular Imaging Data
              </span>
              <p className="text-[11px] text-slate-400">
                Cropped anterior segment photographs, slit-lamp captures, or retinal fundus scans uploaded by the user. Transferred securely in volatile memory solely for inference.
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
              <span className="font-bold text-white text-xs flex items-center gap-2">
                <FileText className="w-4 h-4 text-teal-400" /> Clinical Symptom Indicators
              </span>
              <p className="text-[11px] text-slate-400">
                User-selected phenotypic markers (pain intensity, visual acuity drop, light sensitivity, discharge characteristics, symptom duration, and affected eye lateralization).
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
              <span className="font-bold text-white text-xs flex items-center gap-2">
                <FileCheck className="w-4 h-4 text-indigo-400" /> Optional Systemic Biomarkers
              </span>
              <p className="text-[11px] text-slate-400">
                Non-identifying physiological metrics (approximate patient age group, systolic/diastolic blood pressure, HbA1c percentage, smoking status) used to contextualize ophthalmic risk.
              </p>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-2">
              <span className="font-bold text-white text-xs flex items-center gap-2">
                <Server className="w-4 h-4 text-amber-400" /> Operational & Hardware Telemetry
              </span>
              <p className="text-[11px] text-slate-400">
                System latency, GPU tensor throughput, memory allocation limits, and API error codes to monitor inference reliability and prevent system outages.
              </p>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'processing-purpose',
      icon: <Cpu className="w-5 h-5 text-indigo-400" />,
      badge: 'Inference Pipeline',
      badgeColor: 'bg-indigo-950/80 text-indigo-300 border-indigo-800',
      title: '3. How Data is Processed & AI Assistant Integration',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            Uploaded scans and clinical symptom inputs are processed through our multi-stage AI screening pipeline:
          </p>
          <ol className="space-y-2 list-decimal list-inside pl-1 text-slate-300">
            <li>
              <strong>Vision Ensemble Inference:</strong> Images are normalized and processed across our ensemble models (ConvNeXt-Small, DenseNet-201, EfficientNet-V2, EfficientNet-B4) to calculate probability distributions across 12 ocular conditions.
            </li>
            <li>
              <strong>Explainability Heatmaps (Grad-CAM):</strong> Gradient-weighted Class Activation Mapping calculates visual heatmaps showing the precise anatomical regions influencing model activation.
            </li>
            <li>
              <strong>Uncertainty Quantification:</strong> Monte Carlo dropout calculates epistemic uncertainty intervals to alert clinicians when model confidence is ambiguous.
            </li>
            <li>
              <strong>Clinical Assistant (Gemini 2.0 Flash):</strong> When engaging with the clinical chat assistant, the current diagnostic summary and general symptom inputs are sent ephemerally to Google Gemini API to produce context-aware guidance. <em>Google does not use API customer data to train foundation models.</em>
            </li>
            <li>
              <strong>Standardized Healthcare Records:</strong> Clinicians may export findings in HL7 FHIR R4 JSON standard or tamper-evident PDF reports for electronic health record (EHR) integration.
            </li>
          </ol>
        </div>
      )
    },
    {
      id: 'hipaa-gdpr',
      icon: <Lock className="w-5 h-5 text-emerald-400" />,
      badge: 'Statutory Safeguards',
      badgeColor: 'bg-emerald-950/80 text-emerald-300 border-emerald-800',
      title: '4. HIPAA, GDPR & Patient De-Identification Standards',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            We adhere to the safe-harbor de-identification principles outlined in the <strong>Health Insurance Portability and Accountability Act (HIPAA)</strong> and the European Union <strong>General Data Protection Regulation (GDPR)</strong>:
          </p>
          <div className="space-y-2">
            {[
              {
                title: 'Safe Harbor De-Identification',
                desc: 'Images must not contain full facial photographs, medical record numbers (MRNs), patient names, or hospital barcodes. The integrated client-side cropper facilitates isolating only the relevant ocular structure.'
              },
              {
                title: 'In-Memory Volatile Processing',
                desc: 'Uploaded diagnostic images are processed within volatile GPU/CPU RAM buffers and are not persisted to persistent public cloud disk storage by default.'
              },
              {
                title: 'Configurable Clinical Data Retention',
                desc: 'Audit logs containing clinical predictions are retained solely for session duration or local hospital database synchronization as authorized by clinical system administrators.'
              }
            ].map((item, idx) => (
              <div key={idx} className="p-3 rounded-xl bg-slate-900 border border-slate-800 flex items-start gap-2.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                <div>
                  <strong className="text-white text-xs block">{item.title}</strong>
                  <span className="text-[11px] text-slate-400">{item.desc}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )
    },
    {
      id: 'security-measures',
      icon: <Key className="w-5 h-5 text-amber-400" />,
      badge: 'Defense in Depth',
      badgeColor: 'bg-amber-950/80 text-amber-300 border-amber-800',
      title: '5. Technical Security & Encryption Architecture',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            We implement state-of-the-art cryptographic safeguards and infrastructure isolation:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
            <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1">
              <span className="text-xs font-bold text-amber-300 block">TLS 1.3 Transport</span>
              <p className="text-[11px] text-slate-400">
                All data transmission between the browser client and inference backend is encrypted via HTTPS with modern TLS 1.3 ciphers.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1">
              <span className="text-xs font-bold text-teal-300 block">AES-256 at Rest</span>
              <p className="text-[11px] text-slate-400">
                Any temporary diagnostic cache or local SQLite/MongoDB store utilizes AES-256 standard cryptographic file system protection.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1">
              <span className="text-xs font-bold text-cyan-300 block">Isolated Enclaves</span>
              <p className="text-[11px] text-slate-400">
                Inference backends run in sandboxed container environments with restricted outbound networking privileges.
              </p>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'user-rights',
      icon: <UserCheck className="w-5 h-5 text-purple-400" />,
      badge: 'User Control',
      badgeColor: 'bg-purple-950/80 text-purple-300 border-purple-800',
      title: '6. User Rights, Portability & Data Purge',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            In accordance with GDPR and patient autonomy frameworks, users and clinical administrators maintain comprehensive rights over their data:
          </p>
          <ul className="space-y-2 list-none pl-1">
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-purple-400 shrink-0 mt-0.5" />
              <span><strong>Right to Portability:</strong> Export complete clinical results, heatmaps, and biomarker metadata at any time in FHIR R4 JSON format or formatted PDF reports.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-purple-400 shrink-0 mt-0.5" />
              <span><strong>Right to Immediate Purge:</strong> Clearing the scan or refreshing the session immediately deletes the transient image buffer from client memory.</span>
            </li>
            <li className="flex items-start gap-2">
              <CheckCircle2 className="w-3.5 h-3.5 text-purple-400 shrink-0 mt-0.5" />
              <span><strong>No Persistent Tracking Cookies:</strong> We do not deploy third-party advertising cookies or cross-site tracking beacons.</span>
            </li>
          </ul>
        </div>
      )
    }
  ]

  const filteredSections = activeTab === 'all'
    ? policySections
    : policySections.filter(s => s.id === activeTab)

  return (
    <div className="space-y-10 animate-fade-in max-w-5xl mx-auto">
      {/* Header Banner */}
      <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <button
            onClick={() => onNavigate('diagnostic')}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-cyan-400 hover:text-cyan-300 transition-colors mb-3 group"
          >
            <ArrowLeft className="w-3.5 h-3.5 group-hover:-translate-x-1 transition-transform" />
            Back to Diagnostic Screening
          </button>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-teal-950/80 text-teal-300 border border-teal-800/80 mb-2">
            <Lock className="w-3.5 h-3.5" /> Clinical Data Protection & Privacy Governance
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Privacy Policy
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
            How OphthalmoAI processes, secures, and safeguards clinical imagery, patient biomarkers, and diagnostic inferences.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="glass-panel px-4 py-2.5 rounded-xl border border-slate-800 text-left sm:text-right">
            <span className="text-[10px] text-slate-400 uppercase font-mono block flex items-center gap-1.5 sm:justify-end">
              <Calendar className="w-3 h-3 text-teal-400" /> Effective Date
            </span>
            <span className="text-xs font-bold text-slate-200">September 11, 2026</span>
          </div>
          <button
            onClick={() => onNavigate('terms')}
            className="px-4 py-2.5 text-xs font-semibold text-slate-300 hover:text-white bg-slate-900/90 hover:bg-slate-800 rounded-xl border border-slate-700 transition flex items-center gap-1.5"
          >
            <FileText className="w-3.5 h-3.5 text-cyan-400" /> View Terms & Conditions
          </button>
        </div>
      </div>

      {/* Trust Badges Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
        {[
          { label: 'Zero Ad Monetization', desc: 'No commercial sale of clinical images', color: 'border-emerald-800/60 bg-emerald-950/20 text-emerald-400' },
          { label: 'HIPAA De-Identified', desc: 'Safe harbor compliant intake', color: 'border-cyan-800/60 bg-cyan-950/20 text-cyan-400' },
          { label: 'TLS 1.3 Transport', desc: 'Encrypted client-to-inference socket', color: 'border-teal-800/60 bg-teal-950/20 text-teal-400' },
          { label: 'FHIR R4 Portability', desc: 'Open standard interoperability', color: 'border-indigo-800/60 bg-indigo-950/20 text-indigo-400' },
        ].map((badge, bi) => (
          <div key={bi} className={`p-3.5 rounded-2xl border ${badge.color} space-y-1`}>
            <span className="text-xs font-bold block">{badge.label}</span>
            <span className="text-[10px] text-slate-400 block">{badge.desc}</span>
          </div>
        ))}
      </div>

      {/* Section Filter Pills */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800/80 pb-4">
        <span className="text-[11px] uppercase font-mono text-slate-500 mr-1">Filter Sections:</span>
        <button
          onClick={() => setActiveTab('all')}
          className={`px-3 py-1 rounded-xl text-xs font-semibold border transition ${
            activeTab === 'all'
              ? 'bg-teal-500/20 text-teal-300 border-teal-500/40 shadow-sm'
              : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
          }`}
        >
          All Topics
        </button>
        {policySections.map(s => (
          <button
            key={s.id}
            onClick={() => setActiveTab(s.id)}
            className={`px-3 py-1 rounded-xl text-xs font-semibold border transition ${
              activeTab === s.id
                ? 'bg-teal-500/20 text-teal-300 border-teal-500/40 shadow-sm'
                : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            {s.title.split('.')[1] || s.title}
          </button>
        ))}
      </div>

      {/* Sections Accordion / Cards */}
      <div className="space-y-6">
        {filteredSections.map(sec => (
          <div
            key={sec.id}
            className="glass-panel p-6 sm:p-7 rounded-2xl border border-slate-800 space-y-4 hover:border-slate-700 transition"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/60 pb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-slate-900 border border-slate-800">
                  {sec.icon}
                </div>
                <h2 className="text-base sm:text-lg font-bold text-white tracking-tight">
                  {sec.title}
                </h2>
              </div>
              <span className={`self-start sm:self-auto text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-lg border ${sec.badgeColor}`}>
                {sec.badge}
              </span>
            </div>

            <div>{sec.content}</div>
          </div>
        ))}
      </div>

      {/* Data Protection Officer Contact */}
      <div className="glass-card p-6 sm:p-8 rounded-3xl border border-slate-800 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Mail className="w-4 h-4 text-teal-400" /> Data Protection Officer (DPO) Contact
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              If you have inquiries regarding medical data privacy, HIPAA compliance verification, or wish to exercise data subject deletion rights:
            </p>
            <div className="pt-1 font-mono text-xs text-teal-300">
              dpo-privacy@ophthalmoai.org
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <button
              onClick={() => onNavigate('terms')}
              className="px-5 py-3 rounded-xl text-xs font-semibold bg-slate-900 text-slate-300 hover:text-white border border-slate-700 hover:border-slate-600 transition"
            >
              Review Terms of Service
            </button>
            <button
              onClick={() => onNavigate('diagnostic')}
              className="px-6 py-3 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 shadow-lg shadow-teal-500/20 transition flex items-center justify-center gap-2"
            >
              <Eye className="w-4 h-4" />
              <span>Return to Diagnostic Tool</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
