import { useState, useRef, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  MessageCircle, X, Send, Loader2,
  Bot, User, AlertCircle, Sparkles, ChevronDown, RefreshCw,
  Mic, MicOff, Volume2, VolumeX, Globe, Eye,
} from 'lucide-react'
import DOMPurify from 'dompurify'
import { getActiveApiUrl, FALLBACK_TUNNEL_URL } from './apiConfig'

const MAX_INPUT_LENGTH  = 2000
const MAX_HISTORY_TURNS = 20

const INDIC_LANGUAGES = [
  { code: 'hi-IN', label: 'हिन्दी' },
  { code: 'en-IN', label: 'English' },
  { code: 'bn-IN', label: 'বাংলা' },
  { code: 'ta-IN', label: 'தமிழ்' },
  { code: 'te-IN', label: 'తెలుగు' },
  { code: 'mr-IN', label: 'मराठी' },
  { code: 'gu-IN', label: 'ગુજરાતી' },
  { code: 'kn-IN', label: 'ಕನ್ನಡ' },
  { code: 'ml-IN', label: 'മലയാളം' },
  { code: 'pa-IN', label: 'ਪੰਜਾਬੀ' },
]

const QUICK_QUESTIONS_MAP = {
  'hi-IN': [
    'मेरी जांच के परिणाम का क्या मतलब है?',
    'मोतियाबिंद या ग्लूकोमा के शुरुआती लक्षण क्या हैं?',
    'मुझे तुरंत डॉक्टर को कब दिखाना चाहिए?',
    'आंखों को स्वस्थ रखने के लिए क्या उपाय करें?',
  ],
  'en-IN': [
    'What does my screening result mean?',
    'How does the Tri-Backbone AI Ensemble work?',
    'What are the early signs of cataracts or glaucoma?',
    'When should I see an eye doctor urgently?',
  ],
}

const PURIFY_CONFIG = {
  ALLOWED_TAGS:  [],
  ALLOWED_ATTR:  [],
  KEEP_CONTENT:  true,
}

function sanitise(text) {
  if (!text || typeof text !== 'string') return ''
  return DOMPurify.sanitize(text, PURIFY_CONFIG)
}

function renderMarkdown(rawText) {
  const text  = sanitise(rawText)
  const lines = text.split('\n')
  const nodes = []
  let listBuf = []

  const flushList = () => {
    if (listBuf.length === 0) return
    nodes.push(
      <ul key={`ul-${nodes.length}`} className="list-disc pl-4 space-y-1 my-1.5 text-slate-700">
        {listBuf.map((item, i) => (
          <li key={i} className="text-xs leading-relaxed">{inlineTokens(item)}</li>
        ))}
      </ul>
    )
    listBuf = []
  }

  lines.forEach((line, idx) => {
    const bulletMatch = line.match(/^(?:[-•*]|\d+\.)\s+(.+)/)
    if (bulletMatch) {
      listBuf.push(bulletMatch[1])
      return
    }
    flushList()

    if (line.trim() === '') {
      nodes.push(<br key={`br-${idx}`} />)
      return
    }
    nodes.push(
      <span key={`line-${idx}`} className="block leading-relaxed">
        {inlineTokens(line)}
      </span>
    )
  })

  flushList()
  return <>{nodes}</>
}

function inlineTokens(text) {
  const boldParts = text.split(/\*\*(.+?)\*\*/g)
  return boldParts.flatMap((part, i) => {
    if (i % 2 === 1) {
      return [<strong key={`b-${i}`} className="font-bold text-cyan-900">{inlineItalicCode(part)}</strong>]
    }
    return [inlineItalicCode(part)]
  })
}

function inlineItalicCode(text) {
  const parts = text.split(/`(.+?)`/g)
  return parts.flatMap((part, i) => {
    if (i % 2 === 1) {
      return [
        <code key={`c-${i}`}
          className="font-mono text-[11px] bg-slate-100 text-teal-800 px-1.5 py-0.5 rounded border border-slate-200">
          {part}
        </code>
      ]
    }
    const italicParts = part.split(/\*(.+?)\*/g)
    return italicParts.map((p, j) =>
      j % 2 === 1
        ? <em key={`em-${i}-${j}`} className="italic text-slate-600">{p}</em>
        : <span key={`t-${i}-${j}`}>{p}</span>
    )
  })
}

const TypingDots = () => (
  <div className="flex items-center gap-1.5 px-3 py-2.5">
    {[0, 1, 2].map(i => (
      <span
        key={i}
        className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce"
        style={{ animationDelay: `${i * 0.15}s`, animationDuration: '0.8s' }}
      />
    ))}
  </div>
)

const ChatBot = ({ diagnosisContext }) => {
  const [selectedLanguage, setSelectedLanguage] = useState('hi-IN')
  const [isHighContrast,   setIsHighContrast]   = useState(false)

  const getInitialGreeting = useCallback((lang) => {
    if (lang === 'hi-IN') {
      return diagnosisContext
        ? `नमस्ते! आपकी आंख की जांच में **${diagnosisContext.diagnosis}** के लक्षण (${diagnosisContext.confidence?.toFixed(1)}% सटीकता) मिले हैं। मैं इसके बारे में समझाने और आपके प्रश्नों का उत्तर देने के लिए उपस्थित हूँ। आप बोलकर (माइक) भी पूछ सकते हैं!`
        : "नमस्ते! मैं नेत्र स्वास्थ्य और रेटिनल स्क्रीनिंग सहायक हूँ। अपनी आंखों के लक्षणों के बारे में पूछने के लिए नीचे लिखें या माइक बटन दबाकर बोलें।"
    }
    return diagnosisContext
      ? `Hi! Your scan indicates signs of **${diagnosisContext.diagnosis}** with ${diagnosisContext.confidence?.toFixed(1)}% confidence. I can explain what this means, answer questions about symptoms, or suggest next steps. How can I help?`
      : "Hi there! I'm here to help you understand your eye scan results, discuss symptoms, or answer questions about eye health. What's on your mind?"
  }, [diagnosisContext])

  const [isOpen,          setIsOpen]          = useState(false)
  const [isMinimized,     setIsMinimized]     = useState(false)
  const [messages,        setMessages]        = useState([{ role: 'assistant', content: getInitialGreeting('hi-IN') }])
  const [input,           setInput]           = useState('')
  const [loading,         setLoading]         = useState(false)
  const [inputError,      setInputError]      = useState('')

  // Voice recording & Audio playback states
  const [isRecording,      setIsRecording]      = useState(false)
  const [recordingSeconds, setRecordingSeconds]  = useState(0)
  const [playingMsgIndex,  setPlayingMsgIndex]  = useState(null)

  const messagesEndRef   = useRef(null)
  const inputRef         = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef   = useRef([])
  const audioPlayerRef   = useRef(null)
  const recordTimerRef   = useRef(null)

  useEffect(() => {
    if (isOpen && !isMinimized)
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isOpen, isMinimized])

  useEffect(() => {
    if (isOpen && !isMinimized) inputRef.current?.focus()
  }, [isOpen, isMinimized])

  // Update greeting when diagnosisContext or language changes
  useEffect(() => {
    setMessages([{ role: 'assistant', content: getInitialGreeting(selectedLanguage) }])
  }, [selectedLanguage, getInitialGreeting])

  // Stop recording timer when recording ends
  useEffect(() => {
    if (isRecording) {
      setRecordingSeconds(0)
      recordTimerRef.current = setInterval(() => {
        setRecordingSeconds(s => s + 1)
      }, 1000)
    } else {
      if (recordTimerRef.current) clearInterval(recordTimerRef.current)
      setRecordingSeconds(0)
    }
    return () => {
      if (recordTimerRef.current) clearInterval(recordTimerRef.current)
    }
  }, [isRecording])

  const handleInputChange = useCallback((e) => {
    const val = e.target.value
    setInput(val)
    if (val.length > MAX_INPUT_LENGTH) {
      setInputError(`Message too long (${val.length}/${MAX_INPUT_LENGTH} chars)`)
    } else {
      setInputError('')
    }
  }, [])

  // Audio Playback handler
  const handlePlayAudio = useCallback((msg, idx) => {
    // If already playing this message, stop it
    if (playingMsgIndex === idx) {
      if (audioPlayerRef.current) {
        audioPlayerRef.current.pause()
        audioPlayerRef.current = null
      }
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel()
      }
      setPlayingMsgIndex(null)
      return
    }

    // Stop any existing playback
    if (audioPlayerRef.current) {
      audioPlayerRef.current.pause()
      audioPlayerRef.current = null
    }
    if (window.speechSynthesis) {
      window.speechSynthesis.cancel()
    }

    // Option 1: Native Sarvam synthesized base64 audio
    if (msg.audio_base64) {
      try {
        const audio = new Audio(`data:audio/wav;base64,${msg.audio_base64}`)
        audioPlayerRef.current = audio
        setPlayingMsgIndex(idx)
        audio.onended = () => {
          setPlayingMsgIndex(null)
          audioPlayerRef.current = null
        }
        audio.onerror = () => {
          setPlayingMsgIndex(null)
          audioPlayerRef.current = null
        }
        audio.play().catch(err => {
          console.warn('Audio play failed, falling back to Web Speech:', err)
          fallbackWebSpeech(msg.content, selectedLanguage, idx)
        })
        return
      } catch (e) {
        console.warn('Failed to initialize audio element:', e)
      }
    }

    // Option 2: Browser Web Speech API fallback
    fallbackWebSpeech(msg.content, selectedLanguage, idx)
  }, [playingMsgIndex, selectedLanguage])

  const fallbackWebSpeech = (text, lang, idx) => {
    if (!('speechSynthesis' in window)) return
    const cleanText = text.replace(/[*#`_]/g, '')
    const utterance = new SpeechSynthesisUtterance(cleanText)
    utterance.lang = lang || 'hi-IN'
    utterance.onend = () => setPlayingMsgIndex(null)
    utterance.onerror = () => setPlayingMsgIndex(null)
    setPlayingMsgIndex(idx)
    window.speechSynthesis.speak(utterance)
  }

  // Send message to backend
  const sendMessage = useCallback(async (text) => {
    const messageText = (text || input).trim()
    if (!messageText || loading) return

    if (messageText.length > MAX_INPUT_LENGTH) {
      setInputError(`Message exceeds ${MAX_INPUT_LENGTH} characters`)
      return
    }

    const userMessage = { role: 'user', content: messageText }
    const updatedMessages = [...messages, userMessage]
    setMessages(updatedMessages)
    setInput('')
    setInputError('')
    setLoading(true)

    try {
      const apiUrl = getActiveApiUrl()

      const historyToSend = messages
        .slice(1)
        .slice(-MAX_HISTORY_TURNS)
        .map(m => ({ role: m.role, content: m.content }))

      let res
      const payload = {
        message:                messageText,
        history:                historyToSend,
        diagnosis_context:      diagnosisContext || null,
        language:               selectedLanguage,
        audio_output_requested: true,
      }

      try {
        res = await axios.post(`${apiUrl}/chat`, payload)
      } catch (postErr) {
        if (apiUrl === '/api' && FALLBACK_TUNNEL_URL) {
          res = await axios.post(`${FALLBACK_TUNNEL_URL}/chat`, payload)
        } else {
          throw postErr
        }
      }

      const data = res.data
      const safeReply = sanitise(data.reply || '')
      const newMsg = {
        role:         'assistant',
        content:      safeReply,
        is_emergency: data.is_emergency || false,
        audio_base64: data.audio_base64 || null,
      }
      setMessages(prev => [...prev, newMsg])

      // Auto-play audio if in accessibility / high-contrast mode
      if (data.audio_base64 && isHighContrast) {
        handlePlayAudio(newMsg, updatedMessages.length)
      }
    } catch (err) {
      console.error('Chat error:', err)
      const serverDetail = err?.response?.data?.detail
      const fallback     = `I am currently unable to process your request. Error: ${err.message}. For urgent eye concerns, please contact an ophthalmologist or visit an emergency room immediately.`
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: sanitise(serverDetail || fallback) },
      ])
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages, diagnosisContext, selectedLanguage, isHighContrast, handlePlayAudio])

  // Voice Recording via MediaRecorder + Sarvam STT
  const startRecording = async () => {
    try {
      audioChunksRef.current = []
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' })
        // Release tracks
        stream.getTracks().forEach(track => track.stop())
        await processVoiceBlob(audioBlob)
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (err) {
      console.warn('Microphone access failed, falling back to Web Speech API:', err)
      startWebSpeechRecognition()
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const processVoiceBlob = async (audioBlob) => {
    setLoading(true)
    try {
      const apiUrl = getActiveApiUrl()
      const formData = new FormData()
      formData.append('file', audioBlob, 'patient_voice.webm')
      formData.append('language', selectedLanguage)
      formData.append('audio_output_requested', 'true')
      if (diagnosisContext) {
        formData.append('diagnosis_context_json', JSON.stringify(diagnosisContext))
      }

      const res = await axios.post(`${apiUrl}/chat/voice`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      const data = res.data
      const userText = data.transcript || '🎙️ [Spoken audio query]'
      const safeReply = sanitise(data.reply || '')

      const userMsg = { role: 'user', content: userText }
      const botMsg = {
        role:         'assistant',
        content:      safeReply,
        is_emergency: data.is_emergency || false,
        audio_base64: data.audio_base64 || null,
      }

      setMessages(prev => {
        const nextMsgs = [...prev, userMsg, botMsg]
        // Auto-play synthesized voice guidance for spoken conversation
        if (data.audio_base64) {
          setTimeout(() => handlePlayAudio(botMsg, nextMsgs.length - 1), 300)
        }
        return nextMsgs
      })
    } catch (err) {
      console.error('Voice processing error:', err)
      // If voice endpoint fails, prompt user
      setInputError('Could not process voice recording. Please type your message.')
    } finally {
      setLoading(false)
    }
  }

  const startWebSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('Speech recognition is not supported in this browser. Please type your message.')
      return
    }
    const recognition = new SpeechRecognition()
    recognition.lang = selectedLanguage || 'hi-IN'
    recognition.interimResults = false
    recognition.maxAlternatives = 1

    recognition.onstart = () => setIsRecording(true)
    recognition.onend   = () => setIsRecording(false)
    recognition.onerror = () => setIsRecording(false)
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript
      if (transcript) {
        sendMessage(transcript)
      }
    }
    recognition.start()
  }

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }, [sendMessage])

  const canSend = input.trim().length > 0
    && input.length <= MAX_INPUT_LENGTH
    && !loading

  const quickQuestions = QUICK_QUESTIONS_MAP[selectedLanguage] || QUICK_QUESTIONS_MAP['en-IN']

  return (
    <>
      {/* Floating trigger button */}
      <button
        onClick={() => { setIsOpen(o => !o); setIsMinimized(false) }}
        className="fixed z-50 flex items-center justify-center transition-all duration-300 rounded-full shadow-2xl bottom-6 right-6 w-14 h-14 hover:scale-110 active:scale-95 border border-cyan-500/40"
        style={{
          background: isOpen
            ? 'linear-gradient(135deg, #0F172A, #1E293B)'
            : 'linear-gradient(135deg, #00ADB5, #0891B2)',
          boxShadow: '0 8px 32px rgba(0, 173, 181, 0.45)',
        }}
        aria-label={isOpen ? 'Close AI Doctor chat' : 'Open AI Doctor chat'}
        aria-expanded={isOpen}
      >
        {isOpen
          ? <X className="w-6 h-6 text-white" />
          : <MessageCircle className="w-6 h-6 text-white" />
        }
      </button>

      {/* Floating Chat Modal */}
      {isOpen && (
        <div
          role="dialog"
          aria-label="OphthalmoAI Doctor chat"
          className={`fixed z-50 flex flex-col overflow-hidden border bottom-20 sm:bottom-24 right-4 sm:right-6 rounded-2xl bg-white ${
            isHighContrast ? 'border-amber-400 ring-2 ring-amber-500' : 'border-slate-200'
          }`}
          style={{
            width: 'min(420px, calc(100vw - 32px))',
            height: isMinimized ? 'auto' : 'min(580px, calc(100vh - 110px))',
            boxShadow: '0 20px 45px -10px rgba(15, 23, 42, 0.25), 0 0 20px rgba(2, 132, 199, 0.15)',
          }}
        >
          {/* Chat Header */}
          <div className="flex items-center gap-2.5 px-3.5 py-3 shrink-0 bg-gradient-to-r from-slate-900 via-slate-800 to-cyan-950 border-b border-slate-700">
            <div className="flex items-center justify-center rounded-full w-8 h-8 bg-cyan-500/20 border border-cyan-400/40 shrink-0">
              <Bot className="w-4 h-4 text-cyan-400" />
            </div>

            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <p className="text-xs font-bold text-white tracking-wide truncate">OphthalmoAI Clinical</p>
                <span className="px-1 py-0.2 text-[8px] font-bold uppercase tracking-wider rounded bg-cyan-950 text-cyan-300 border border-cyan-700 shrink-0">
                  Sarvam+Gemini
                </span>
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shrink-0" />
                <p className="text-[10px] text-cyan-200/90 truncate">
                  {diagnosisContext ? `Diagnosis: ${diagnosisContext.diagnosis}` : 'Regional Outreach Triage'}
                </p>
              </div>
            </div>

            {/* Language Selector */}
            <div className="flex items-center gap-1 shrink-0">
              <Globe className="w-3.5 h-3.5 text-cyan-300" />
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="bg-slate-800/90 text-cyan-200 text-[10px] font-medium border border-slate-600 rounded px-1.5 py-1 outline-none cursor-pointer hover:border-cyan-400"
                aria-label="Select Indic Language"
              >
                {INDIC_LANGUAGES.map(lang => (
                  <option key={lang.code} value={lang.code} className="bg-slate-900 text-white">
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>

            {/* High Contrast / Accessibility Toggle */}
            <button
              onClick={() => setIsHighContrast(c => !c)}
              className={`p-1.5 rounded transition ${
                isHighContrast ? 'bg-amber-500 text-black font-bold' : 'hover:bg-slate-800 text-slate-300 hover:text-white'
              }`}
              title="Toggle Large Text / High Contrast for Low Vision"
              aria-label="High contrast mode"
            >
              <Eye className="w-3.5 h-3.5" />
            </button>

            {/* Minimize toggle */}
            <button
              onClick={() => setIsMinimized(m => !m)}
              className="p-1 transition rounded-lg hover:bg-slate-800 text-slate-300 hover:text-white"
              aria-label={isMinimized ? 'Expand chat' : 'Minimise chat'}
            >
              <ChevronDown className={`w-4 h-4 transition-transform ${isMinimized ? 'rotate-180' : ''}`} />
            </button>
          </div>

          {!isMinimized && (
            <>
              {/* Clinical Educational Disclaimer Banner */}
              <div className="flex items-start gap-2 px-3 py-1.5 shrink-0 bg-amber-50 border-b border-amber-200">
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 text-amber-600 shrink-0" />
                <p className="text-[10px] text-amber-900 leading-tight font-medium">
                  {selectedLanguage === 'hi-IN'
                    ? 'शैक्षणिक और प्राथमिक ट्राइएज हेतु। कृपया नेत्र चिकित्सक से प्रत्यक्ष जांच कराएं।'
                    : 'For educational triage only. Consult an eye specialist for binding clinical diagnosis.'}
                </p>
              </div>

              {/* Messages Scroll Area */}
              <div
                className={`flex-1 p-3.5 space-y-3 overflow-y-auto ${
                  isHighContrast ? 'bg-slate-900 text-white' : 'bg-slate-50/70 text-slate-800'
                }`}
                aria-live="polite"
              >
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex items-end gap-2 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                  >
                    <div
                      className={`flex items-center justify-center rounded-full w-7 h-7 shrink-0 ${
                        msg.role === 'assistant'
                          ? 'bg-cyan-100 text-cyan-700 border border-cyan-200'
                          : 'bg-indigo-100 text-indigo-700 border border-indigo-200'
                      }`}
                    >
                      {msg.role === 'assistant'
                        ? <Bot className="w-4 h-4" />
                        : <User className="w-4 h-4" />
                      }
                    </div>

                    <div
                      className={`max-w-[85%] px-3.5 py-2.5 rounded-2xl ${
                        isHighContrast ? 'text-sm font-semibold' : 'text-xs'
                      } leading-relaxed ${
                        msg.is_emergency
                          ? 'bg-red-50 text-red-950 border-2 border-red-500 rounded-bl-sm shadow-md'
                          : msg.role === 'assistant'
                          ? isHighContrast
                            ? 'bg-black text-white border-2 border-amber-400 rounded-bl-sm'
                            : 'bg-white text-slate-800 border border-slate-200 rounded-bl-sm shadow-2xs'
                          : 'bg-gradient-to-r from-cyan-600 to-teal-600 text-white rounded-br-sm shadow-2xs font-medium'
                      }`}
                    >
                      {renderMarkdown(msg.content)}

                      {/* Speaker Audio Playback Button for Assistant */}
                      {msg.role === 'assistant' && (
                        <div className="mt-2 pt-1.5 border-t border-slate-100 dark:border-slate-800 flex items-center justify-between">
                          <button
                            onClick={() => handlePlayAudio(msg, i)}
                            className={`flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-md font-medium transition ${
                              playingMsgIndex === i
                                ? 'bg-cyan-600 text-white animate-pulse'
                                : 'bg-slate-100 hover:bg-cyan-50 text-slate-700 hover:text-cyan-800'
                            }`}
                            aria-label="Listen to voice guidance"
                          >
                            {playingMsgIndex === i ? (
                              <>
                                <VolumeX className="w-3 h-3" />
                                <span>Stop Audio</span>
                              </>
                            ) : (
                              <>
                                <Volume2 className="w-3 h-3 text-cyan-600" />
                                <span>सुनें / Listen</span>
                              </>
                            )}
                          </button>
                          {msg.audio_base64 && (
                            <span className="text-[9px] text-slate-400">Sarvam Bulbul TTS</span>
                          )}
                        </div>
                      )}

                      {/* Indian Emergency Escalation Links */}
                      {msg.is_emergency && (
                        <div className="mt-2.5 pt-2 border-t border-red-200">
                          <span className="block text-[10px] font-bold uppercase text-red-700 mb-1.5">
                            🚨 आपातकालीन सेवा / Urgent Emergency
                          </span>
                          <div className="flex gap-2">
                            <a
                              href="tel:112"
                              className="px-2.5 py-1 bg-red-600 hover:bg-red-500 text-white rounded font-bold text-[11px] shadow-xs"
                            >
                              Call 112 (National)
                            </a>
                            <a
                              href="tel:108"
                              className="px-2.5 py-1 bg-amber-600 hover:bg-amber-500 text-white rounded font-bold text-[11px] shadow-xs"
                            >
                              Ambulance 108
                            </a>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex items-end gap-2">
                    <div className="flex items-center justify-center rounded-full w-7 h-7 shrink-0 bg-cyan-100 text-cyan-700 border border-cyan-200">
                      <Bot className="w-4 h-4" />
                    </div>
                    <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-sm shadow-2xs px-3 py-2">
                      <TypingDots />
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Quick Questions Strip */}
              {messages.length <= 2 && !loading && (
                <div className="px-3 py-2 shrink-0 bg-slate-50 border-t border-slate-200/80">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Sparkles className="w-3 h-3 text-cyan-600" />
                    <span className="text-[9px] font-bold uppercase tracking-wider text-slate-500">
                      सुझाए गए प्रश्न / Suggested Questions
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {quickQuestions.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => sendMessage(q)}
                        className="text-[10px] px-2.5 py-1 rounded-full border border-slate-200 bg-white text-slate-700 hover:text-cyan-800 hover:border-cyan-300 hover:bg-cyan-50 transition shadow-2xs font-medium text-left"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Input Area with Mic & Speech Support */}
              <div className="p-2.5 shrink-0 bg-white border-t border-slate-200">
                <div className="flex items-end gap-1.5">
                  <div className="flex flex-col flex-1">
                    <textarea
                      ref={inputRef}
                      value={input}
                      onChange={handleInputChange}
                      onKeyDown={handleKeyDown}
                      placeholder={
                        isRecording
                          ? 'बोल रहे हैं... सुन रहे हैं / Listening...'
                          : selectedLanguage === 'hi-IN'
                          ? 'लक्षण लिखें या माइक बटन दबाकर बोलें...'
                          : 'Describe eye symptoms or speak using the mic...'
                      }
                      disabled={isRecording}
                      className={`flex-1 text-xs px-3 py-2 rounded-xl resize-none outline-none border transition-all ${
                        isRecording
                          ? 'bg-red-50/60 border-red-300 text-red-900 placeholder-red-400 animate-pulse'
                          : 'bg-slate-50 border-slate-200 text-slate-800 placeholder-slate-400 focus:bg-white focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/15'
                      }`}
                      style={{
                        borderColor: inputError ? '#EF4444' : undefined,
                        maxHeight: '75px',
                        minHeight: '38px',
                      }}
                      aria-label="Chat message input"
                    />
                    {inputError && (
                      <p className="text-[10px] text-red-600 mt-0.5 px-1 font-medium">
                        {inputError}
                      </p>
                    )}
                  </div>

                  {/* Microphone Button (Voice-first accessibility) */}
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={loading}
                    className={`flex items-center justify-center w-9 h-9 transition-all rounded-xl shrink-0 shadow-sm ${
                      isRecording
                        ? 'bg-red-600 hover:bg-red-500 text-white animate-bounce'
                        : 'bg-slate-100 hover:bg-cyan-50 border border-slate-300 text-slate-700 hover:text-cyan-700'
                    }`}
                    title={isRecording ? 'Stop & Send Recording' : 'Speak symptoms in your regional language'}
                    aria-label={isRecording ? 'Stop recording' : 'Start voice recording'}
                  >
                    {isRecording ? (
                      <span className="flex items-center gap-0.5 text-[9px] font-bold">
                        <MicOff className="w-4 h-4 text-white" />
                        <span className="text-[9px]">{recordingSeconds}s</span>
                      </span>
                    ) : (
                      <Mic className="w-4 h-4" />
                    )}
                  </button>

                  {/* Send Button */}
                  <button
                    onClick={() => sendMessage()}
                    disabled={!canSend || isRecording}
                    className="flex items-center justify-center w-9 h-9 transition-all rounded-xl bg-gradient-to-br from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed shrink-0 shadow-sm"
                    aria-label="Send message"
                  >
                    {loading
                      ? <Loader2 className="w-4 h-4 text-white animate-spin" />
                      : <Send className="w-4 h-4 text-white" />
                    }
                  </button>
                </div>

                <div className="flex items-center justify-between text-[9px] mt-1.5 px-1 text-slate-400">
                  <span>🇮🇳 Sarvam Indic AI (Saaras + Bulbul)</span>
                  <span>Emergency: 112 / 108</span>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </>
  )
}

export default ChatBot
