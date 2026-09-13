import { useState, useRef, useEffect, useCallback } from 'react'
import axios from 'axios'
import {
  MessageCircle, X, Send, Loader2,
  Bot, User, AlertCircle, Sparkles, ChevronDown, RefreshCw,
} from 'lucide-react'
import DOMPurify from 'dompurify'
import { getActiveApiUrl, FALLBACK_TUNNEL_URL } from './App'

const MAX_INPUT_LENGTH  = 2000
const MAX_HISTORY_TURNS = 20
const QUICK_QUESTIONS   = [
  'What does my screening result mean?',
  'How does the Tri-Backbone AI Ensemble work?',
  'What are the early signs of cataracts or glaucoma?',
  'When should I see an eye doctor urgently?',
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
  const initialMessage = diagnosisContext
    ? `Hi! Your scan indicates signs of **${diagnosisContext.diagnosis}** with ${diagnosisContext.confidence?.toFixed(1)}% confidence. I can explain what this means, answer any questions about your symptoms, or suggest next steps. How can I help?`
    : "Hi there! I'm here to help you understand your eye scan results, discuss symptoms, or answer questions about eye health. What's on your mind?"

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
      const apiUrl = getActiveApiUrl()

      const historyToSend = messages
        .slice(1)
        .slice(-MAX_HISTORY_TURNS)
        .map(m => ({ role: m.role, content: m.content }))

      let res
      try {
        res = await axios.post(`${apiUrl}/chat`, {
          message:           messageText,
          history:           historyToSend,
          diagnosis_context: diagnosisContext || null,
        })
      } catch (postErr) {
        if (apiUrl === '/api' && FALLBACK_TUNNEL_URL) {
          res = await axios.post(`${FALLBACK_TUNNEL_URL}/chat`, {
            message:           messageText,
            history:           historyToSend,
            diagnosis_context: diagnosisContext || null,
          })
        } else {
          throw postErr
        }
      }

      const data = res.data
      const safeReply = sanitise(data.reply || '')
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: safeReply,
        is_emergency: data.is_emergency || false,
      }])
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

      {/* Floating Chat Modal */}
      {isOpen && (
        <div
          role="dialog"
          aria-label="OphthalmoAI Doctor chat"
          className="fixed z-50 flex flex-col overflow-hidden border border-slate-200 bottom-20 sm:bottom-24 right-4 sm:right-6 rounded-2xl bg-white"
          style={{
            width: 'min(400px, calc(100vw - 32px))',
            height: isMinimized ? 'auto' : 'min(560px, calc(100vh - 120px))',
            boxShadow: '0 20px 45px -10px rgba(15, 23, 42, 0.25), 0 0 20px rgba(2, 132, 199, 0.1)',
          }}
        >
          {/* Chat Header */}
          <div className="flex items-center gap-3 px-4 py-3.5 shrink-0 bg-gradient-to-r from-slate-900 via-slate-800 to-cyan-950 border-b border-slate-700">
            <div className="flex items-center justify-center rounded-full w-9 h-9 bg-cyan-500/20 border border-cyan-400/40">
              <Bot className="w-5 h-5 text-cyan-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <p className="text-sm font-bold text-white tracking-wide">OphthalmoAI Assistant</p>
                <span className="px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-widest rounded bg-cyan-950 text-cyan-300 border border-cyan-700">
                  Clinical AI
                </span>
              </div>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                <p className="text-[11px] text-cyan-200/90 truncate">
                  {diagnosisContext ? `Context: ${diagnosisContext.diagnosis}` : 'Eye Health Guidance'}
                </p>
              </div>
            </div>
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
              <div className="flex items-start gap-2 px-3.5 py-2 shrink-0 bg-amber-50 border-b border-amber-200">
                <AlertCircle className="w-4 h-4 mt-0.5 text-amber-600 shrink-0" />
                <p className="text-[11px] text-amber-800 leading-tight font-medium">
                  For educational screening use only. Consult an eye care professional for medical diagnosis.
                </p>
              </div>

              {/* Messages Scroll Area */}
              <div
                className="flex-1 p-4 space-y-3.5 overflow-y-auto bg-slate-50/70"
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
                      className={`max-w-[84%] px-3.5 py-2.5 rounded-2xl text-xs leading-relaxed ${
                        msg.is_emergency
                          ? 'bg-red-50 text-red-900 border border-red-300 rounded-bl-sm shadow-sm'
                          : msg.role === 'assistant'
                          ? 'bg-white text-slate-800 border border-slate-200 rounded-bl-sm shadow-sm'
                          : 'bg-gradient-to-r from-cyan-600 to-teal-600 text-white rounded-br-sm shadow-sm font-medium'
                      }`}
                    >
                      {renderMarkdown(msg.content)}
                      {msg.is_emergency && (
                        <div className="mt-2.5 pt-2 border-t border-red-200 flex items-center justify-between">
                          <span className="text-[10px] font-bold uppercase text-red-700">Urgent: Seek Emergency Care</span>
                          <a href="tel:911" className="px-2 py-1 bg-red-600 hover:bg-red-500 text-white rounded font-bold text-[10px]">Call 911</a>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="flex items-end gap-2.5">
                    <div className="flex items-center justify-center rounded-full w-7 h-7 shrink-0 bg-cyan-100 text-cyan-700 border border-cyan-200">
                      <Bot className="w-4 h-4" />
                    </div>
                    <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-sm shadow-sm px-3 py-2">
                      <TypingDots />
                    </div>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>

              {/* Quick Questions Strip */}
              {messages.length <= 2 && !loading && (
                <div className="px-3.5 py-2 shrink-0 bg-slate-50 border-t border-slate-200/80">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <Sparkles className="w-3 h-3 text-cyan-600" />
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                      Suggested Questions
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {QUICK_QUESTIONS.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => sendMessage(q)}
                        className="text-[11px] px-2.5 py-1 rounded-full border border-slate-200 bg-white text-slate-700 hover:text-cyan-800 hover:border-cyan-300 hover:bg-cyan-50 transition-all duration-150 shadow-2xs font-medium text-left"
                      >
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Input Area */}
              <div className="p-3 shrink-0 bg-white border-t border-slate-200">
                <div className="flex gap-2">
                  <div className="flex flex-col flex-1">
                    <textarea
                      ref={inputRef}
                      value={input}
                      onChange={handleInputChange}
                      onKeyDown={handleKeyDown}
                      placeholder="Ask about eye health, symptoms, prevention..."
                      className="flex-1 text-xs px-3 py-2.5 rounded-xl resize-none outline-none bg-slate-50 border border-slate-200 text-slate-800 placeholder-slate-400 focus:bg-white focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/15 transition-all"
                      style={{
                        borderColor: inputError ? '#EF4444' : undefined,
                        maxHeight: '80px',
                      }}
                      aria-label="Chat message input"
                    />
                    {inputError && (
                      <p className="text-[10px] text-red-600 mt-0.5 px-1 font-medium">
                        {inputError}
                      </p>
                    )}
                  </div>
                  <button
                    onClick={() => sendMessage()}
                    disabled={!canSend}
                    className="flex items-center self-start justify-center w-10 h-10 transition-all rounded-xl bg-gradient-to-br from-cyan-600 to-teal-600 hover:from-cyan-500 hover:to-teal-500 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed shrink-0 shadow-md shadow-cyan-600/20"
                    aria-label="Send message"
                  >
                    {loading
                      ? <Loader2 className="w-4 h-4 text-white animate-spin" />
                      : <Send className="w-4 h-4 text-white" />
                    }
                  </button>
                </div>
                <p className="text-center text-[9px] mt-2 text-slate-400">
                  Medical AI Assistant · Educational guidance only
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
