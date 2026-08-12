import { useState, useRef, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  MessageCircle, X, Send, Loader2,
  Bot, User, AlertCircle, Sparkles, ChevronDown, RefreshCw,
} from 'lucide-react'
import DOMPurify from 'dompurify'

const MAX_INPUT_LENGTH  = 2000
const MAX_HISTORY_TURNS = 20
const QUICK_QUESTIONS   = [
  'What are the early signs of cataracts?',
  'How can I prevent eye disease?',
  'When should I see an eye doctor urgently?',
  'What does my screening result mean?',
]

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
      <ul key={`ul-${nodes.length}`} className="list-disc pl-4 space-y-1 my-1.5 text-slate-300">
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
      return [<strong key={`b-${i}`} className="font-semibold text-cyan-300">{inlineItalicCode(part)}</strong>]
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
          className="font-mono text-[11px] bg-slate-800 text-teal-300 px-1.5 py-0.5 rounded border border-slate-700">
          {part}
        </code>
      ]
    }
    const italicParts = part.split(/\*(.+?)\*/g)
    return italicParts.map((p, j) =>
      j % 2 === 1
        ? <em key={`em-${i}-${j}`} className="italic text-slate-200">{p}</em>
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
  const initialMessage = diagnosisContext
    ? `Hello! I see your AI screening detected **${diagnosisContext.diagnosis}** with ${diagnosisContext.confidence?.toFixed(1)}% confidence. I am here to explain this result, answer symptoms questions, and offer guidance on next steps. What would you like to ask?`
    : "Hello! I am OphthalmoAI Doctor, your AI eye health assistant powered by Google Gemini. How can I assist you with your eye health today?"

  const [isOpen,      setIsOpen]      = useState(false)
  const [isMinimized, setIsMinimized] = useState(false)
  const [messages,    setMessages]    = useState([{ role: 'assistant', content: initialMessage }])
  const [input,       setInput]       = useState('')
  const [loading,     setLoading]     = useState(false)
  const [inputError,  setInputError]  = useState('')

  const messagesEndRef = useRef(null)
  const inputRef       = useRef(null)

  useEffect(() => {
    if (isOpen && !isMinimized)
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isOpen, isMinimized])

  useEffect(() => {
    if (isOpen && !isMinimized) inputRef.current?.focus()
  }, [isOpen, isMinimized])

  useEffect(() => {
    if (diagnosisContext) {
      setMessages([{
        role: 'assistant',
        content: `Hello! I see your AI screening detected **${diagnosisContext.diagnosis}** with ${diagnosisContext.confidence?.toFixed(1)}% confidence. What would you like to know about this condition?`,
      }])
    }
  }, [diagnosisContext])

  const handleInputChange = useCallback((e) => {
    const val = e.target.value
    setInput(val)
    if (val.length > MAX_INPUT_LENGTH) {
      setInputError(`Message too long (${val.length}/${MAX_INPUT_LENGTH} chars)`)
    } else {
      setInputError('')
    }
  }, [])

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
      const apiUrl = import.meta.env.VITE_API_URL || '/api'

      const historyToSend = messages
        .slice(1)
        .slice(-MAX_HISTORY_TURNS)
        .map(m => ({ role: m.role, content: m.content }))

      const { data } = await axios.post(`${apiUrl}/chat`, {
        message:           messageText,
        history:           historyToSend,
        diagnosis_context: diagnosisContext || null,
      })

      const safeReply = sanitise(data.reply || '')
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: safeReply,
        is_emergency: data.is_emergency || false,
      }])
    } catch (err) {
      const serverDetail = err?.response?.data?.detail
      const fallback     = 'I am currently unable to process your request. For urgent eye concerns, please contact an ophthalmologist or visit an emergency room immediately.'
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: sanitise(serverDetail || fallback) },
      ])
    } finally {
      setLoading(false)
    }
  }, [input, loading, messages, diagnosisContext])

  const handleKeyDown = useCallback((e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }, [sendMessage])

  const canSend = input.trim().length > 0
    && input.length <= MAX_INPUT_LENGTH
    && !loading

  return (
    <>
      {}
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

      {}
      {isOpen && (
        <div
          role="dialog"
          aria-label="OphthalmoAI Doctor chat"
          className="fixed z-50 flex flex-col overflow-hidden border border-slate-700/60 bottom-24 right-6 rounded-2xl glass-panel"
          style={{
            width: 'min(400px, calc(100vw - 32px))',
            height: isMinimized ? 'auto' : '560px',
            boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7), 0 0 30px rgba(0, 173, 181, 0.2)',
          }}
        >
          {}
          <div className="flex items-center gap-3 px-4 py-3.5 shrink-0 bg-gradient-to-r from-slate-950 via-slate-900 to-cyan-950/80 border-b border-slate-800">
            <div className="flex items-center justify-center rounded-full w-9 h-9 bg-cyan-500/20 border border-cyan-500/40">
              <Bot className="w-5 h-5 text-cyan-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <p className="text-sm font-bold text-white tracking-wide">OphthalmoAI Doctor</p>
                <span className="px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-widest rounded bg-cyan-950 text-cyan-400 border border-cyan-800">
                  Gemini Free Tier
                </span>
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <p className="text-[11px] text-cyan-200/80 truncate">
                  {diagnosisContext ? `Context: ${diagnosisContext.diagnosis}` : 'AI Eye Health Specialist'}
                </p>
              </div>
            </div>
            <button
              onClick={() => setIsMinimized(m => !m)}
              className="p-1 transition rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white"
              aria-label={isMinimized ? 'Expand chat' : 'Minimise chat'}
            >
              <ChevronDown className={`w-4 h-4 transition-transform ${isMinimized ? 'rotate-180' : ''}`} />
            </button>
          </div>

          {!isMinimized && (
            <>
              {}
              <div className="flex items-start gap-2 px-3.5 py-2 shrink-0 bg-amber-950/40 border-b border-amber-900/40">
                <AlertCircle className="w-4 h-4 mt-0.5 text-amber-400 shrink-0" />
                <p className="text-[11px] text-amber-200/90 leading-tight">
                  Educational AI guidance — not a substitute for professional clinical medical evaluation.
                </p>
              </div>

              {}
              <div
                className="flex-1 p-4 space-y-3.5 overflow-y-auto bg-slate-950/60"
                aria-live="polite"
              >
                {messages.map((msg, i) => (
                  <div
                    key={i}
                    className={`flex items-end gap-2.5 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                  >
                    <div
                      className={`flex items-center justify-center rounded-full w-7 h-7 shrink-0 ${
                        msg.role === 'assistant'
                          ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                          : 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/30'
                      }`}
                    >
                      {msg.role === 'assistant'
                        ? <Bot className="w-4 h-4" />
                        : <User className="w-4 h-4" />
                      }
                    </div>
                    <div
                      className={`max-w-[82%] px-3.5 py-2.5 rounded-2xl text-xs ${
                        msg.is_emergency
                          ? 'bg-red-950/90 text-red-100 border border-red-700 rounded-bl-sm shadow-xl animate-pulse-glow'
                          : msg.role === 'assistant'
                          ? 'bg-slate-900/90 text-slate-100 border border-slate-800 rounded-bl-sm shadow-md'
                          : 'bg-gradient-to-r from-cyan-600 to-teal-600 text-white rounded-br-sm shadow-md'
                      }`}
                    >
                      {renderMarkdown(msg.content)}
                      {msg.is_emergency && (
                        <div className="mt-2.5 pt-2 border-t border-red-800/80 flex items-center justify-between">
                          <span className="text-[10px] font-bold uppercase text-red-300">Call Emergency: 911 / 112</span>
                          <a href="tel:911" className="px-2 py-1 bg-red-600 hover:bg-red-500 text-white rounded font-bold text-[10px]">Call Now</a>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex items-end gap-2.5">
                    <div className="flex items-center justify-center rounded-full w-7 h-7 shrink-0 bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                      <Bot className="w-4 h-4" />
                    </div>
                    <div className="bg-slate-900 border border-slate-800 rounded-2xl rounded-bl-sm">
                      <TypingDots />
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {}
              {messages.length <= 2 && !loading && (
                <div className="px-3.5 pb-2 shrink-0 bg-slate-950/60">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Sparkles className="w-3 h-3 text-cyan-400" />
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                      Suggested Questions
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {QUICK_QUESTIONS.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => sendMessage(q)}
                        className="text-[11px] px-2.5 py-1 rounded-full border border-slate-700 bg-slate-900/80 text-slate-300 hover:text-white hover:border-cyan-500/60 hover:bg-cyan-950/50 transition-all duration-200"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {}
              <div className="p-3 shrink-0 bg-slate-900 border-t border-slate-800">
                <div className="flex gap-2">
                  <div className="flex flex-col flex-1">
                    <textarea
                      ref={inputRef}
                      value={input}
                      onChange={handleInputChange}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask about eye health, symptoms, prevention..."
                      className="flex-1 text-xs px-3 py-2.5 rounded-xl resize-none outline-none glass-input"
                      style={{
                        borderColor: inputError ? '#EF4444' : undefined,
                        maxHeight: '80px',
                      }}
                      aria-label="Chat message input"
                    />
                    {inputError && (
                      <p className="text-[10px] text-red-400 mt-0.5 px-1">
                        {inputError}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={() => sendMessage()}
                    disabled={!canSend}
                    className="flex items-center self-start justify-center w-10 h-10 transition-all rounded-xl bg-gradient-to-br from-cyan-500 to-teal-600 hover:from-cyan-400 hover:to-teal-500 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed shrink-0 shadow-lg shadow-cyan-500/20"
                    aria-label="Send message"
                  >
                    {loading
                      ? <Loader2 className="w-4 h-4 text-white animate-spin" />
                      : <Send className="w-4 h-4 text-white" />
                    }
                  </button>
                </div>
                <p className="text-center text-[9px] mt-2 text-slate-500">
                  Powered by Google Gemini Free Tier API · Press Enter to send
                </p>
              </div>
            </>
          )}
        </div>
      )}
    </>
  )
}

export default ChatBot
