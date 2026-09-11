import React, { useState } from 'react'
import {
  FileText, ShieldAlert, AlertTriangle, CheckCircle2,
  Scale, Lock, Eye, ArrowLeft, Stethoscope, Cpu,
  HelpCircle, ChevronRight, Sparkles, Building2,
  Mail, Calendar, ExternalLink, ShieldCheck
} from 'lucide-react'

export default function TermsPage({ onNavigate }) {
  const [activeSection, setActiveSection] = useState('all')

  const sections = [
    {
      id: 'medical-disclaimer',
      icon: <ShieldAlert className="w-5 h-5 text-rose-400" />,
      badge: 'Critical Notice',
      badgeColor: 'bg-rose-950/80 text-rose-300 border-rose-800',
      title: '1. Medical AI & Clinical Use Disclaimer',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <div className="p-4 rounded-xl bg-rose-950/30 border border-rose-800/60 text-rose-200 space-y-2">
            <div className="flex items-center gap-2 font-bold text-sm text-rose-300">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              Auxiliary Decision-Support Only: Not Autonomous Medical Care
            </div>
            <p>
              OphthalmoAI is an assistive artificial intelligence screening and research tool engineered to assist qualified healthcare professionals and clinical researchers in ocular image evaluation. <strong>It is NOT an FDA-cleared autonomous diagnostic device, nor is it a replacement for a board-certified ophthalmologist, optometrist, or emergency physician.</strong>
            </p>
          </div>
          <p>
            The machine learning inferences, probabilistic predictions, Grad-CAM attention maps, and Monte Carlo confidence intervals provided by this platform are algorithmic estimates derived from trained convolutional neural network ensembles (including ConvNeXt-Small, DenseNet-201, EfficientNet-V2, and EfficientNet-B4). They must always be clinically correlated with patient medical history, physical slit-lamp examination, optical coherence tomography (OCT), visual field testing, and direct biomicroscopy.
          </p>
          <div className="p-3.5 rounded-xl bg-amber-950/20 border border-amber-900/40 text-amber-200">
            <span className="font-bold text-amber-300 block mb-1">Emergency Situations</span>
            If a patient is presenting with acute severe ocular pain, sudden onset of profound vision loss, mechanical trauma, chemical eye burn, flashes with showers of dark floaters, or signs of acute angle-closure glaucoma, <strong>do not rely on software screening</strong>. Immediately direct the patient to an emergency ophthalmic facility.
          </div>
        </div>
      )
    },
    {
      id: 'acceptable-use',
      icon: <Stethoscope className="w-5 h-5 text-cyan-400" />,
      badge: 'Permitted Usage',
      badgeColor: 'bg-cyan-950/80 text-cyan-300 border-cyan-800',
      title: '2. Acceptable Use & User Qualifications',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            By accessing or operating the OphthalmoAI platform, you certify that you meet the prerequisite qualifications and agree to use the software solely for legitimate clinical, educational, epidemiological, or research screening purposes:
          </p>
          <ul className="space-y-2 list-none pl-1">
            {[
              'You are a licensed clinician, optometrist, ophthalmology resident, triage nurse, or medical researcher operating under clinical supervision.',
              'You will not use this platform to conduct unauthorized medical practice or offer definitive diagnosis without clinical verification.',
              'You will not perform automated high-frequency stress tests, scrape model predictions, or attempt denial-of-service against the API gateway.',
              'You will not reverse-engineer, decompile, extract proprietary weights, or poison dataset caches for adversarial purposes.'
            ].map((rule, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-cyan-400 shrink-0 mt-0.5" />
                <span>{rule}</span>
              </li>
            ))}
          </ul>
        </div>
      )
    },
    {
      id: 'patient-consent',
      icon: <Lock className="w-5 h-5 text-teal-400" />,
      badge: 'Compliance',
      badgeColor: 'bg-teal-950/80 text-teal-300 border-teal-800',
      title: '3. Patient Consent, HIPAA & De-Identification',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            Protecting patient privacy is a paramount ethical and statutory mandate. When uploading external anterior segment or retinal fundus photographs:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="font-semibold text-white block">De-Identification Mandate</span>
              <p className="text-[11px] text-slate-400">
                You warrant that any uploaded eye scan or biometric metadata has been stripped of direct Protected Health Information (PHI) including full patient names, national ID numbers, and facial biometrics outside the cropped ocular region.
              </p>
            </div>
            <div className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
              <span className="font-semibold text-white block">Patient Informed Consent</span>
              <p className="text-[11px] text-slate-400">
                You affirm that appropriate institutional review board (IRB) approval, hospital consent protocol, or patient informed consent has been obtained prior to submitting clinical imaging for computational screening.
              </p>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'intellectual-property',
      icon: <Cpu className="w-5 h-5 text-indigo-400" />,
      badge: 'Proprietary Rights',
      badgeColor: 'bg-indigo-950/80 text-indigo-300 border-indigo-800',
      title: '4. Intellectual Property & Model Weights',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            The OphthalmoAI platform, including but not limited to its Meta-Classifier ensemble architecture, model weights, Grad-CAM visual heatmapping pipeline, FHIR R4 mapping algorithms, user interface source code, brand assets, and technical documentation are the proprietary intellectual property of the OphthalmoAI project authors and contributors.
          </p>
          <p>
            Users retain full ownership of their original diagnostic image inputs and exported patient reports. By submitting imagery for inference, you grant OphthalmoAI a transient, non-exclusive, royalty-free license solely to execute requested neural network inference, Grad-CAM visualization, and temporary session display.
          </p>
        </div>
      )
    },
    {
      id: 'liability',
      icon: <Scale className="w-5 h-5 text-amber-400" />,
      badge: 'Legal Disclaimer',
      badgeColor: 'bg-amber-950/80 text-amber-300 border-amber-800',
      title: '5. Limitation of Liability & "As-Is" Provision',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p className="uppercase tracking-wide font-mono text-[11px] text-slate-400">
            Disclaimer of Warranties
          </p>
          <p>
            The platform is provided strictly on an <strong>&quot;AS-IS&quot;</strong> and <strong>&quot;AS-AVAILABLE&quot;</strong> basis, without warranties of any kind, whether express, implied, statutory, or otherwise, including but not limited to warranties of merchantability, fitness for a particular clinical diagnostic purpose, title, or non-infringement.
          </p>
          <div className="p-3.5 rounded-xl bg-slate-900 border border-slate-800 text-slate-300 space-y-1">
            <span className="font-semibold text-white block">Limitation on Damages</span>
            <p className="text-[11px] text-slate-400">
              To the fullest extent permitted under applicable law, in no event shall the authors, maintainers, affiliated institutions, or contributors be held liable for any direct, indirect, incidental, special, consequential, or punitive damages (including medical malpractice claims, misdiagnosis, delayed treatment, loss of life, or clinical downtime) arising out of or in connection with the use or inability to use this platform.
            </p>
          </div>
        </div>
      )
    },
    {
      id: 'updates-governing',
      icon: <Building2 className="w-5 h-5 text-purple-400" />,
      badge: 'Governance',
      badgeColor: 'bg-purple-950/80 text-purple-300 border-purple-800',
      title: '6. Platform Updates, Revisions & Governing Law',
      content: (
        <div className="space-y-3 text-xs text-slate-300 leading-relaxed">
          <p>
            We reserve the right to modify, refine, or retrain the AI models, adjust detection thresholds, or update these Terms and Conditions at any time to reflect advancing medical standards, regulatory guidance, or algorithmic improvements. Continued use of the platform following the posting of revised terms constitutes binding acceptance of those modifications.
          </p>
          <p>
            These terms are governed by and construed in accordance with prevailing computational health software standards and applicable jurisdiction regulations, without giving effect to any conflict of law principles.
          </p>
        </div>
      )
    }
  ]

  const filteredSections = activeSection === 'all'
    ? sections
    : sections.filter(s => s.id === activeSection)

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
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-950/80 text-cyan-300 border border-cyan-800/80 mb-2">
            <Scale className="w-3.5 h-3.5" /> Clinical Legal Terms & User Agreement
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight">
            Terms & Conditions
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
            Please read these clinical and computational terms of service carefully before utilizing the OphthalmoAI screening platform.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="glass-panel px-4 py-2.5 rounded-xl border border-slate-800 text-left sm:text-right">
            <span className="text-[10px] text-slate-400 uppercase font-mono block flex items-center gap-1.5 sm:justify-end">
              <Calendar className="w-3 h-3 text-cyan-400" /> Effective Date
            </span>
            <span className="text-xs font-bold text-slate-200">September 11, 2026</span>
          </div>
          <button
            onClick={() => onNavigate('privacy')}
            className="px-4 py-2.5 text-xs font-semibold text-slate-300 hover:text-white bg-slate-900/90 hover:bg-slate-800 rounded-xl border border-slate-700 transition flex items-center gap-1.5"
          >
            <Lock className="w-3.5 h-3.5 text-teal-400" /> View Privacy Policy
          </button>
        </div>
      </div>

      {/* Quick Summary Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="glass-card p-4 rounded-2xl border border-slate-800 space-y-1.5">
          <div className="flex items-center gap-2 text-rose-400">
            <ShieldAlert className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Clinical Triage Only</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-normal">
            Auxiliary AI software intended to assist clinicians, not replace certified medical evaluation or emergency interventions.
          </p>
        </div>

        <div className="glass-card p-4 rounded-2xl border border-slate-800 space-y-1.5">
          <div className="flex items-center gap-2 text-teal-400">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">HIPAA Alignment</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-normal">
            Uploaded images must be de-identified. We do not store unprotected personal health identifiers by default.
          </p>
        </div>

        <div className="glass-card p-4 rounded-2xl border border-slate-800 space-y-1.5">
          <div className="flex items-center gap-2 text-cyan-400">
            <Cpu className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Meta-Classifier Ensemble</span>
          </div>
          <p className="text-[11px] text-slate-400 leading-normal">
            Multi-backbone deep learning with Monte Carlo epistemic uncertainty estimations for balanced evidence.
          </p>
        </div>
      </div>

      {/* Section Quick Jump Filter */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-800/80 pb-4">
        <span className="text-[11px] uppercase font-mono text-slate-500 mr-1">Filter Sections:</span>
        <button
          onClick={() => setActiveSection('all')}
          className={`px-3 py-1 rounded-xl text-xs font-semibold border transition ${
            activeSection === 'all'
              ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-sm'
              : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
          }`}
        >
          All Sections
        </button>
        {sections.map(s => (
          <button
            key={s.id}
            onClick={() => setActiveSection(s.id)}
            className={`px-3 py-1 rounded-xl text-xs font-semibold border transition ${
              activeSection === s.id
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-sm'
                : 'bg-slate-900/60 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            {s.title.split('.')[1] || s.title}
          </button>
        ))}
      </div>

      {/* Main Sections Stack */}
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

      {/* Contact and Agreement Actions */}
      <div className="glass-card p-6 sm:p-8 rounded-3xl border border-slate-800 space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Mail className="w-4 h-4 text-cyan-400" /> Clinical & Legal Inquiries
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              For regulatory oversight, institutional licensing, academic collaborations, or IRB compliance inquiries, contact our clinical governance desk:
            </p>
            <div className="pt-1 font-mono text-xs text-cyan-300">
              <a href="mailto:akashkundu1152@gmail.com" className="hover:underline text-cyan-300">
                akashkundu1152@gmail.com
              </a>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <button
              onClick={() => onNavigate('privacy')}
              className="px-5 py-3 rounded-xl text-xs font-semibold bg-slate-900 text-slate-300 hover:text-white border border-slate-700 hover:border-slate-600 transition"
            >
              Read Privacy Policy
            </button>
            <button
              onClick={() => onNavigate('diagnostic')}
              className="px-6 py-3 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 shadow-lg shadow-cyan-500/20 transition flex items-center justify-center gap-2"
            >
              <Eye className="w-4 h-4" />
              <span>Acknowledge & Open Screening Tool</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
