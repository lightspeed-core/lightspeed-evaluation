import { useState, useEffect, useRef } from 'react'

function timeAgo(ts) {
  const sec = Math.floor((Date.now() - ts) / 1000)
  if (sec < 60) return `${sec}s ago`
  const min = Math.floor(sec / 60)
  if (min < 60) return `${min}m ago`
  const hr = Math.floor(min / 60)
  return `${hr}h ${min % 60}m ago`
}

export default function RunPage() {
  const [systemConfig, setSystemConfig] = useState('')
  const [apiKeySet, setApiKeySet] = useState(false)
  const [tags, setTags] = useState([])
  const [selectedTag, setSelectedTag] = useState('__all__')
  const [loading, setLoading] = useState(true)
  const [starting, setStarting] = useState(false)

  const [runs, setRuns] = useState([])
  const [selectedRunId, setSelectedRunId] = useState(null)
  const [output, setOutput] = useState('')
  const [exitCode, setExitCode] = useState(null)
  const [fullscreen, setFullscreen] = useState(false)

  const termRef = useRef(null)
  const abortRef = useRef(null)
  const outputRef = useRef('')

  useEffect(() => {
    fetch('/api/run-config')
      .then(r => r.json())
      .then(data => {
        setSystemConfig(data.systemConfig || '')
        setApiKeySet(!!data.apiKey)
        setTags(data.tags || [])
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  useEffect(() => {
    const poll = () => {
      fetch('/api/running-evals')
        .then(r => r.json())
        .then(setRuns)
        .catch(() => {})
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!selectedRunId) return
    if (abortRef.current) abortRef.current.abort()

    outputRef.current = ''
    setOutput('')
    setExitCode(null)

    const controller = new AbortController()
    abortRef.current = controller

    const connect = async () => {
      try {
        const res = await fetch(`/api/eval-stream/${selectedRunId}`, {
          signal: controller.signal,
        })
        if (!res.ok || !res.body) return

        const reader = res.body.getReader()
        const decoder = new TextDecoder()
        let buffer = ''
        let currentEvent = ''

        const processLines = (text) => {
          const lines = text.split('\n')
          buffer = lines.pop()
          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEvent = line.slice(7)
            } else if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                if (currentEvent === 'output') {
                  outputRef.current += data.text
                  setOutput(outputRef.current)
                } else if (currentEvent === 'exit') {
                  setExitCode(data.code)
                }
              } catch { /* skip malformed */ }
              currentEvent = ''
            }
          }
        }

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          processLines(buffer + decoder.decode(value, { stream: true }))
        }
        if (buffer.trim()) processLines(buffer + '\n')
      } catch (err) {
        if (err.name !== 'AbortError') {
          outputRef.current += `\nConnection error: ${err.message}\n`
          setOutput(outputRef.current)
        }
      }
    }
    connect()

    return () => { controller.abort() }
  }, [selectedRunId])

  useEffect(() => {
    if (termRef.current) {
      termRef.current.scrollTop = termRef.current.scrollHeight
    }
  }, [output])

  const tagLabel = (t) => t === '__all__' ? 'all' : t
  const ready = !!(systemConfig && apiKeySet)
  const tagRunning = selectedTag && runs.some(r => r.tag === tagLabel(selectedTag) && r.status === 'running')
  const canStart = ready && !starting && !tagRunning
  const command = selectedTag === '__all__'
    ? `API_KEY=*** lightspeed-eval --system-config ${systemConfig || '${LS_EVAL_SYSTEM_CFG_PATH}'} --eval-data eval.yaml --output-dir results`
    : `API_KEY=*** lightspeed-eval --system-config ${systemConfig || '${LS_EVAL_SYSTEM_CFG_PATH}'} --eval-data eval.yaml --tags ${selectedTag} --output-dir results`

  const startRun = async () => {
    setStarting(true)
    try {
      const res = await fetch('/api/run-eval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ systemConfig, tag: selectedTag }),
      })
      const { id } = await res.json()
      setSelectedRunId(id)
      const runsRes = await fetch('/api/running-evals')
      setRuns(await runsRes.json())
    } catch { /* ignore */ }
    setStarting(false)
  }

  const stopRun = async (e, id) => {
    e.stopPropagation()
    try {
      await fetch(`/api/stop-eval/${id}`, { method: 'POST' })
      const runsRes = await fetch('/api/running-evals')
      setRuns(await runsRes.json())
    } catch { /* ignore */ }
  }

  const selectedRun = runs.find(r => r.id === selectedRunId)

  if (loading) {
    return (
      <div className="explorer-loading">
        <div className="spinner" />
        <p>Loading run configuration...</p>
      </div>
    )
  }

  return (
    <div className="run-page">
      <div className="run-card">
        <div className="run-header">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          <h3>Run Evaluation</h3>
        </div>

        <div className="run-field">
          <label>Evaluation Tag</label>
          <select
            value={selectedTag}
            onChange={e => setSelectedTag(e.target.value)}
          >
            <option value="__all__">All (run everything)</option>
            {tags.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>

        <div className="run-field">
          <label>The following command will be executed</label>
          <pre className="run-command">{command}</pre>
        </div>

        <div className="run-disclaimer">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span><code>LS_EVAL_DATA_PATH</code>, <code>LS_EVAL_SYSTEM_CFG_PATH</code>, <code>LS_EVAL_REPORTS_PATH</code> and <code>API_KEY</code> must be set as environment variables when starting the dev server.</span>
        </div>

        {tagRunning && (
          <div className="run-disclaimer run-disclaimer-warn">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
              <line x1="12" y1="9" x2="12" y2="13" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            <span>Tag <code>{tagLabel(selectedTag)}</code> is already running. Wait for it to finish or stop it before starting a new run.</span>
          </div>
        )}

        <button
          className={`run-start-btn${canStart ? '' : ' disabled'}`}
          disabled={!canStart}
          onClick={startRun}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="5 3 19 12 5 21 5 3" />
          </svg>
          {starting ? 'Starting...' : 'Start'}
        </button>
      </div>

      <div className="run-card">
        <div className="run-header">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
          <h3>Running Evaluations</h3>
        </div>

        {runs.length === 0 ? (
          <p className="run-no-evals">No evaluation pipelines detected. Start one above or run lightspeed-eval from the command line.</p>
        ) : (
          <div className="run-eval-list">
            {runs.map(run => (
              <div
                key={run.id}
                className={`run-eval-item${selectedRunId === run.id ? ' selected' : ''}`}
                onClick={() => setSelectedRunId(run.id)}
              >
                <div className="run-eval-info">
                  <span className="run-eval-conversation">{run.tag === 'all' ? 'All' : run.tag}</span>
                  <span className="run-eval-pid">PID {run.pid}</span>
                  {run.startTime && <span className="run-eval-time">{timeAgo(run.startTime)}</span>}
                  {run.source === 'external' && <span className="run-eval-badge">external</span>}
                </div>
                <div className="run-eval-actions">
                  {run.status === 'running' ? (
                    <>
                      <span className="run-eval-status running">running</span>
                      {run.source === 'web' && (
                        <button className="run-eval-stop" onClick={(e) => stopRun(e, run.id)}>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <rect x="6" y="6" width="12" height="12" rx="2" />
                          </svg>
                          Stop
                        </button>
                      )}
                    </>
                  ) : (
                    <span className={`run-eval-status ${run.exitCode === 0 ? 'success' : 'failure'}`}>
                      exit {run.exitCode}
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {selectedRunId && (
        <div className={`run-terminal-card${fullscreen ? ' fullscreen' : ''}`}>
          <div className="run-terminal-header">
            <span className="run-terminal-title">
              Output{selectedRun ? ` — ${selectedRun.tag === 'all' ? 'All' : selectedRun.tag}` : ''}
            </span>
            {selectedRun?.status === 'running' && <div className="spinner-sm" />}
            {exitCode !== null && (
              <span className={`run-exit-code ${exitCode === 0 ? 'success' : 'failure'}`}>
                exit {exitCode}
              </span>
            )}
            <button
              className="run-terminal-fullscreen"
              onClick={() => {
                const blob = new Blob([output], { type: 'text/plain' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                const d = new Date(selectedRun?.startTime || Date.now())
                const ts = `${d.getFullYear()}${String(d.getMonth()+1).padStart(2,'0')}${String(d.getDate()).padStart(2,'0')}_${String(d.getHours()).padStart(2,'0')}${String(d.getMinutes()).padStart(2,'0')}${String(d.getSeconds()).padStart(2,'0')}`
                a.download = `eval-${selectedRun?.tag || selectedRunId}_${ts}.txt`
                a.click()
                URL.revokeObjectURL(url)
              }}
              title="Download log"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </button>
            <button
              className="run-terminal-fullscreen"
              onClick={() => setFullscreen(f => !f)}
              title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
            >
              {fullscreen ? (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="4 14 10 14 10 20" />
                  <polyline points="20 10 14 10 14 4" />
                  <line x1="14" y1="10" x2="21" y2="3" />
                  <line x1="3" y1="21" x2="10" y2="14" />
                </svg>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="15 3 21 3 21 9" />
                  <polyline points="9 21 3 21 3 15" />
                  <line x1="21" y1="3" x2="14" y2="10" />
                  <line x1="3" y1="21" x2="10" y2="14" />
                </svg>
              )}
            </button>
          </div>
          <pre className="run-terminal" ref={termRef}>{output || 'Waiting for output...\n'}</pre>
        </div>
      )}
    </div>
  )
}
