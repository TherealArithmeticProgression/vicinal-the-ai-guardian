import { useState, useRef, useEffect } from 'react'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastGuard, setLastGuard] = useState(null)
  const chatEndRef = useRef(null)

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || loading) return

    const userMessage = { role: 'user', content: input }
    setMessages(prev => [...prev, userMessage])
    setInput('')
    setLoading(true)
    setLastGuard(null)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMessage],
          session_id: 'demo-session',
        }),
      })

      const data = await res.json()
      setLastGuard(data.guard)

      if (data.blocked) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: 'Vicinal Guard Intervention: Prompt blocked. ' + data.guard.reason,
          isBlocked: true
        }])
      } else {
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: data.reply
        }])
      }
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error connecting to backend: ${err.message}`,
        isBlocked: true
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="layout">
      {/* LEFT: Chat UI */}
      <div className="panel">
        <div className="chat-window">
          {messages.length === 0 && (
            <div style={{ textAlign: 'center', margin: 'auto', color: 'var(--text-muted)' }}>
              <h3>Start a conversation</h3>
              <p>Prompts will be evaluated by Vicinal before reaching the model.</p>
              <p style={{ marginTop: '1rem', fontSize: '0.9rem' }}>
                Try telling it to "Ignore previous instructions", or ask for "PII data".
              </p>
            </div>
          )}
          
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role} ${msg.isBlocked ? 'blocked' : ''}`}>
              {msg.role === 'assistant' && <><strong style={{ color: 'inherit' }}>{msg.isBlocked ? '🛡️ Guard' : '🤖 Assistant'}</strong><br/></>}
              {msg.role === 'user' && <><strong style={{ color: 'inherit' }}>You</strong><br/></>}
              <div style={{ marginTop: '0.5rem', whiteSpace: 'pre-wrap' }}>
                {msg.content}
              </div>
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="chat-input-container">
          <form onSubmit={handleSubmit} className="chat-input-form">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask anything..."
              disabled={loading}
              autoFocus
            />
            <button type="submit" disabled={loading || !input.trim()}>
              {loading ? <div className="loader" /> : 'Send'}
            </button>
          </form>
        </div>
      </div>

      {/* RIGHT: Vicinal Telemetry */}
      <div className="panel guard-panel">
        <div className="guard-header">
          <h2>
            🛡️ Vicinal Engine
            <span className="live-badge">Live</span>
          </h2>
        </div>

        {lastGuard ? (
          <>
            <div className={`verdict-box verdict-${lastGuard.verdict}`}>
              <div className="verdict-title">{lastGuard.verdict}</div>
              <div className="score-display">
                Threat Score: <strong>{(lastGuard.composite_score * 100).toFixed(1)}%</strong>
              </div>
              <div className="reason-text">{lastGuard.reason}</div>
            </div>

            <div className="evaluator-list">
              <h3 style={{ fontSize: '1rem', margin: '1rem 0 0.5rem', color: 'var(--text-muted)' }}>
                Evaluator Telemetry
              </h3>
              
              {lastGuard.evaluators.map((ev, i) => (
                <div key={i} className="evaluator-card">
                  <div className="evaluator-header">
                    <strong>{ev.name.replace('_', ' ').toUpperCase()}</strong>
                    <span>{ev.latency_ms.toFixed(1)} ms</span>
                  </div>
                  <div style={{ fontSize: '0.85rem' }}>
                    Score: {(ev.score * 100).toFixed(1)}%
                  </div>
                  
                  {ev.hits.length > 0 && (
                    <div style={{ marginTop: '0.5rem' }}>
                      {ev.hits.map((hit, j) => (
                        <div key={j} className="hit-item">
                          <strong style={{ color: 'var(--brand-orange)' }}>[{hit.category}]</strong><br/>
                          {hit.evidence}
                        </div>
                      ))}
                    </div>
                  )}
                  {ev.hits.length === 0 && (
                    <div className="hit-item" style={{ borderLeftColor: 'var(--brand-green)' }}>
                      No threat hits detected.
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div style={{ display: 'grid', placeItems: 'center', height: '100%', color: 'var(--border)' }}>
            <div style={{ textAlign: 'center' }}>
               <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🛡️</div>
               Waiting for payload...
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
