import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

const API = 'http://localhost:8000'

const SUGGESTIONS = [
  "What does my music taste say about me?",
  "Recommend something energetic from the index",
  "Why was Song_04 recommended to me?",
  "What makes a track sound calm vs energetic?",
]

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const bottomRef = useRef()

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const send = async (text) => {
    if (!text.trim()) return
    const userMsg = { role: 'user', content: text }
    const newMessages = [...messages, userMsg]
    setMessages(newMessages)
    setInput('')
    setLoading(true)

    try {
      const res = await axios.post(`${API}/chat`, {
        messages: newMessages.map(m => ({ role: m.role, content: m.content })),
      })
      setMessages([...newMessages, { role: 'assistant', content: res.data.response }])
    } catch (e) {
      const detail = e.response?.data?.detail
      let errMsg = 'Sorry, something went wrong.'
      if (typeof detail === 'string' && detail.includes('credit')) {
        errMsg = '⚠️ API credits needed. Add credits at console.anthropic.com to use the chat.'
      } else if (detail) {
        errMsg = `Error: ${detail}`
      }
      setMessages([...newMessages, { role: 'assistant', content: errMsg }])
    } finally {
      setLoading(false)
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(input) }
  }

  return (
    <div className="chat-container">
      <div style={{ marginBottom: 24 }}>
        <h1 className="page-title">Chat</h1>
        <p className="page-subtitle">Ask about your music taste, get recommendations, or explore acoustic features.</p>
      </div>

      {messages.length === 0 && (
        <div className="chat-suggestions">
          {SUGGESTIONS.map(s => (
            <button key={s} className="suggestion-chip" onClick={() => send(s)}>{s}</button>
          ))}
        </div>
      )}

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-message ${msg.role}`}>
            <div className="chat-avatar">
              {msg.role === 'user' ? '◉' : '◈'}
            </div>
            <div className="chat-bubble">{msg.content}</div>
          </div>
        ))}
        {loading && (
          <div className="chat-message assistant">
            <div className="chat-avatar">◈</div>
            <div className="chat-bubble">
              <div className="loading" style={{ padding: 0 }}>
                <div className="spinner" /> Thinking...
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          className="chat-input"
          placeholder="Ask Auralis anything about your music..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
        />
        <button className="btn btn-primary" onClick={() => send(input)} disabled={!input.trim() || loading}>
          ↑
        </button>
        {messages.length > 0 && (
          <button className="btn btn-ghost" onClick={() => setMessages([])} title="Clear chat">✕</button>
        )}
      </div>
    </div>
  )
}
