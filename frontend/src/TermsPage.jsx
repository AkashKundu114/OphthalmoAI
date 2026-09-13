import React, { useState } from 'react'
import {
  FileText, ShieldAlert, AlertTriangle, CheckCircle2,
  Scale, Lock, Eye, ArrowLeft, Stethoscope, Cpu,
  HelpCircle, ChevronRight, Sparkles, Building2,
  Mail, ExternalLink, ShieldCheck
} from 'lucide-react'

export default function TermsPage({ onNavigate }) {
  const [activeSection, setActiveSection] = useState('all')

  const sections = [
    {
      id: 'medical-disclaimer',
      icon: <ShieldAlert className="w-5 h-5 text-rose-600" />,
      badge: 'Critical Notice',
      badgeColor: 'bg-rose-50 text-rose-700 border-rose-200',
      title: '1. Medical AI & Clinical Use Disclaimer',
      content: (
        <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
          <div className="p-4 rounded-xl bg-rose-50/90 border border-rose-200 text-rose-900 space-y-2">
            <div className="flex items-center gap-2 font-bold text-sm text-rose-800">
              <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0" />
              Auxiliary Decision-Support Only: Not Autonomous Medical Care
            </div>
            <p>
              OphthalmoAI is an assistive artificial intelligence screening and research tool engineered to assist qualified healthcare professionals and clinical researchers in ocular image evaluation. <strong>It is NOT an FDA-cleared autonomous diagnostic device, nor is it a replacement for a board-certified ophthalmologist, optometrist, or emergency physician.</strong>
            </p>
          </div>
          <p>
            The machine learning inferences, probabilistic predictions, Grad-CAM attention maps, and Monte Carlo confidence intervals provided by this platform are algorithmic estimates derived from trained convolutional neural network ensembles (including ConvNeXt-Small, DenseNet-201, EfficientNet-V2, and EfficientNet-B4). They must always be clinically correlated with patient medical history, physical slit-lamp examination, optical coherence tomography (OCT), visual field testing, and direct biomicroscopy.
          </p>
          <div className="p-3.5 rounded-xl bg-amber-50 border border-amber-200 text-amber-900">
            <span className="font-bold text-amber-800 block mb-1">Emergency Situations</span>
            If a patient is presenting with acute severe ocular pain, sudden onset of profound vision loss, mechanical trauma, chemical eye burn, flashes with showers of dark floaters, or signs of acute angle-closure glaucoma, <strong>do not rely on software screening</strong>. Immediately direct the patient to an emergency ophthalmic facility.
          </div>
        </div>
      )
    },
    {
      id: 'acceptable-use',
      icon: <Stethoscope className="w-5 h-5 text-cyan-600" />,
      badge: 'Permitted Usage',
      badgeColor: 'bg-cyan-50 text-cyan-700 border-cyan-200',
      title: '2. Acceptable Use & User Qualifications',
      content: (
        <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
          <p>
            By accessing or operating the OphthalmoAI platform, you agree to use the software solely for legitimate personal health screening, educational, clinical decision-support, or medical research purposes:
          </p>
          <ul className="space-y-2 list-none pl-1">
            {[
              'You may use this tool as a member of the public, patient, caregiver, student, or healthcare professional for preliminary eye health screening and informational purposes.',
              'You understand that this platform provides auxiliary educational screening and does not constitute a legally binding or definitive medical diagnosis.',
              'You will not perform automated high-frequency stress tests, scrape model predictions, or attempt denial-of-service against the API gateway.',
              'You will not reverse-engineer, decompile, extract proprietary model weights, or poison dataset caches for adversarial purposes.'
            ].map((rule, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-cyan-600 shrink-0 mt-0.5" />
                <span>{rule}</span>
              </li>
            ))}
          </ul>
        </div>
      )
    },
    {
      id: 'patient-consent',
      icon: <Lock className="w-5 h-5 text-teal-600" />,
      badge: 'Compliance',
      badgeColor: 'bg-teal-50 text-teal-700 border-teal-200',
      title: '3. Patient Consent, HIPAA & De-Identification',
      content: (
        <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
          <p>
            Protecting patient privacy is a paramount ethical and statutory mandate. When uploading external anterior segment or retinal fundus photographs:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
              <span className="font-semibold text-slate-900 block">De-Identification Mandate</span>
              <p className="text-[11px] text-slate-600">
                You warrant that any uploaded eye scan or biometric metadata has been stripped of direct Protected Health Information (PHI) including full patient names, national ID numbers, and facial biometrics outside the cropped ocular region.
              </p>
            </div>
            <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 space-y-1">
              <span className="font-semibold text-slate-900 block">Patient Informed Consent</span>
              <p className="text-[11px] text-slate-600">
                You affirm that appropriate institutional review board (IRB) approval, hospital consent protocol, or patient informed consent has been obtained prior to submitting clinical imaging for computational screening.
              </p>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'intellectual-property',
      icon: <Cpu className="w-5 h-5 text-indigo-600" />,
      badge: 'Proprietary Rights',
      badgeColor: 'bg-indigo-50 text-indigo-700 border-indigo-200',
      title: '4. Intellectual Property & Model Weights',
      content: (
        <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
          <p>
            The OphthalmoAI platform, including but not limited to its Calibrated Tri-Backbone Vision Ensemble architecture (DenseNet-201, ConvNeXt-Small, EfficientNet-V2-M, EfficientNet-B4), model weights, Grad-CAM visual heatmapping pipeline, FHIR R4 mapping algorithms, user interface source code, brand assets, and technical documentation are the proprietary intellectual property of the OphthalmoAI project authors and contributors.
          </p>
          <p>
            Users retain full ownership of their original diagnostic image inputs and exported patient reports. By submitting imagery for inference, you grant OphthalmoAI a transient, non-exclusive, royalty-free license solely to execute requested neural network inference, Grad-CAM visualization, and temporary session display.
          </p>
        </div>
      )
    },
    {
      id: 'liability',
      icon: <Scale className="w-5 h-5 text-amber-600" />,
      badge: 'Legal Disclaimer',
      badgeColor: 'bg-amber-50 text-amber-700 border-amber-200',
      title: '5. Limitation of Liability & "As-Is" Provision',
      content: (
        <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
          <p className="uppercase tracking-wide font-mono text-[11px] text-slate-500 font-semibold">
            Disclaimer of Warranties
          </p>
          <p>
            The platform is provided strictly on an <strong>&quot;AS-IS&quot;</strong> and <strong>&quot;AS-AVAILABLE&quot;</strong> basis, without warranties of any kind, whether express, implied, statutory, or otherwise, including but not limited to warranties of merchantability, fitness for a particular clinical diagnostic purpose, title, or non-infringement.
          </p>
          <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200 text-slate-700 space-y-1">
            <span className="font-semibold text-slate-900 block">Limitation on Damages</span>
            <p className="text-[11px] text-slate-600">
              To the fullest extent permitted under applicable law, in no event shall the authors, maintainers, affiliated institutions, or contributors be held liable for any direct, indirect, incidental, special, consequential, or punitive damages (including medical malpractice claims, misdiagnosis, delayed treatment, loss of life, or clinical downtime) arising out of or in connection with the use or inability to use this platform.
            </p>
          </div>
        </div>
      )
    },
    {
      id: 'updates-governing',
      icon: <Building2 className="w-5 h-5 text-purple-600" />,
      badge: 'Governance',
      badgeColor: 'bg-purple-50 text-purple-700 border-purple-200',
      title: '6. Platform Updates, Revisions & Governing Law',
      content: (
        <div className="space-y-3 text-xs text-slate-600 leading-relaxed">
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
    <div className="space-y-8 animate-fade-in max-w-5xl mx-auto px-1 sm:px-2">
      {/* Header Banner */}
      <div className="border-b border-slate-200 pb-6 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <button
            onClick={() => onNavigate('diagnostic')}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-cyan-700 hover:text-cyan-800 transition-colors mb-3 group"
          >
            <ArrowLeft className="w-3.5 h-3.5 group-hover:-translate-x-1 transition-transform" />
            Back to Diagnostic Screening
          </button>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-50 text-cyan-800 border border-cyan-200 mb-2">
            <Scale className="w-3.5 h-3.5 text-cyan-600" /> Clinical Legal Terms & User Agreement
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Terms & Conditions
          </h1>
          <p className="text-xs sm:text-sm text-slate-600 mt-1 max-w-2xl leading-relaxed">
            Please read these clinical and computational terms of service carefully before utilizing the OphthalmoAI screening platform.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => onNavigate('privacy')}
            className="px-4 py-2.5 text-xs font-semibold text-slate-700 hover:text-slate-900 bg-white hover:bg-slate-50 rounded-xl border border-slate-200 shadow-2xs transition flex items-center gap-1.5"
          >
            <Lock className="w-3.5 h-3.5 text-teal-600" /> View Privacy Policy
          </button>
        </div>
      </div>

      {/* Quick Summary Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-2xs space-y-1.5">
          <div className="flex items-center gap-2 text-rose-600">
            <ShieldAlert className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Clinical Triage Only</span>
          </div>
          <p className="text-[11px] text-slate-600 leading-normal">
            Auxiliary AI software intended to assist clinicians, not replace certified medical evaluation or emergency interventions.
          </p>
        </div>

        <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-2xs space-y-1.5">
          <div className="flex items-center gap-2 text-teal-600">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">HIPAA Alignment</span>
          </div>
          <p className="text-[11px] text-slate-600 leading-normal">
            Uploaded images must be de-identified. We do not store unprotected personal health identifiers by default.
          </p>
        </div>

        <div className="bg-white p-4 rounded-2xl border border-slate-200 shadow-2xs space-y-1.5">
          <div className="flex items-center gap-2 text-cyan-600">
            <Cpu className="w-4 h-4" />
            <span className="text-xs font-bold uppercase tracking-wider">Calibrated Vision Ensemble</span>
          </div>
          <p className="text-[11px] text-slate-600 leading-normal">
            Multi-backbone deep learning (85.18% SOTA accuracy) with Monte Carlo epistemic uncertainty estimations for balanced clinical evidence.
          </p>
        </div>
      </div>

      {/* Section Quick Jump Filter */}
      <div className="flex flex-wrap items-center gap-2 border-b border-slate-200 pb-4">
        <span className="text-[11px] uppercase font-mono text-slate-500 mr-1 font-medium">Filter Sections:</span>
        <button
          onClick={() => setActiveSection('all')}
          className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition ${
            activeSection === 'all'
              ? 'bg-cyan-50 text-cyan-800 border-cyan-300 shadow-2xs'
              : 'bg-white text-slate-600 border-slate-200 hover:text-slate-900 hover:bg-slate-50'
          }`}
        >
          All Sections
        </button>
        {sections.map(s => (
          <button
            key={s.id}
            onClick={() => setActiveSection(s.id)}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold border transition ${
              activeSection === s.id
                ? 'bg-cyan-50 text-cyan-800 border-cyan-300 shadow-2xs'
                : 'bg-white text-slate-600 border-slate-200 hover:text-slate-900 hover:bg-slate-50'
            }`}
          >
            {s.title.split('.')[1] || s.title}
          </button>
        ))}
      </div>

      {/* Main Sections Stack */}
      <div className="space-y-5">
        {filteredSections.map(sec => (
          <div
            key={sec.id}
            className="bg-white p-6 sm:p-7 rounded-2xl border border-slate-200 shadow-2xs space-y-4 hover:border-slate-300 transition"
          >
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-slate-50 border border-slate-200">
                  {sec.icon}
                </div>
                <h2 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
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
      <div className="bg-white p-6 sm:p-8 rounded-3xl border border-slate-200 shadow-2xs space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div className="space-y-1">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Mail className="w-4 h-4 text-cyan-600" /> Clinical & Legal Inquiries
            </h3>
            <p className="text-xs text-slate-600 leading-relaxed">
              For regulatory oversight, institutional licensing, academic collaborations, or IRB compliance inquiries, contact our clinical governance desk:
            </p>
            <div className="pt-1 font-mono text-xs text-cyan-700">
              <a href="mailto:akashkundu1152@gmail.com" className="hover:underline text-cyan-700 font-medium">
                akashkundu1152@gmail.com
              </a>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3">
            <button
              onClick={() => onNavigate('privacy')}
              className="px-5 py-3 rounded-xl text-xs font-semibold bg-slate-50 text-slate-700 hover:text-slate-900 border border-slate-200 hover:bg-slate-100 transition"
            >
              Read Privacy Policy
            </button>
            <button
              onClick={() => onNavigate('diagnostic')}
              className="px-6 py-3 rounded-xl text-xs font-bold text-white bg-gradient-to-r from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 shadow-md shadow-cyan-600/20 transition flex items-center justify-center gap-2"
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
