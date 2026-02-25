import Markdown from 'react-markdown'

const MD_PATTERN = /[*_`#\-|>\[\]]/

export default function DetailModal({ date, metric, entries, onClose }) {
  const matching = metric
    ? entries.filter(e => e.date === date && e.metric === metric)
    : entries.filter(e => e.date === date)

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Evaluation Details</h2>
            <div className="modal-subtitle">
              {new Date(date).toLocaleString()}
            </div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div className="modal-body">
          {matching.map((e, i) => (
            <div key={i} className="detail-card">
              <div className="detail-row">
                <div className="detail-field">
                  <span className="detail-label">Conversation</span>
                  <span className="detail-value">{e.conversationGroupId}</span>
                </div>
                <div className="detail-field">
                  <span className="detail-label">Turn</span>
                  <span className="detail-value">{e.turnId}</span>
                </div>
                <div className="detail-field">
                  <span className="detail-label">Metric</span>
                  <strong className="detail-value">{e.metric}</strong>
                </div>
              </div>
              <div className="detail-row">
                <div className="detail-field">
                  <span className="detail-label">Score</span>
                  <strong className="detail-value">{e.score !== null ? e.score.toFixed(4) : '-'}</strong>
                </div>
                <div className="detail-field">
                  <span className="detail-label">Result</span>
                  <span className={`badge ${e.result.toLowerCase()}`}>{e.result}</span>
                </div>
                <div className="detail-field">
                  <span className="detail-label">Exec Time</span>
                  <span className="detail-value">{e.executionTime !== null ? e.executionTime.toFixed(3) + 's' : '-'}</span>
                </div>
              </div>
              {e.reason && (
                <div className="detail-field full-width">
                  <span className="detail-label">Reason</span>
                  {MD_PATTERN.test(e.reason) ? (
                    <div className="detail-value detail-reason detail-md"><Markdown>{e.reason}</Markdown></div>
                  ) : (
                    <span className="detail-value detail-reason">{e.reason}</span>
                  )}
                </div>
              )}
              {e.query && (
                <div className="detail-field full-width">
                  <span className="detail-label">Query</span>
                  <pre className="detail-pre">{e.query}</pre>
                </div>
              )}
              {e.response && (
                <div className="detail-field full-width">
                  <span className="detail-label">Response</span>
                  <div className="detail-value detail-response detail-md">
                    <Markdown>{e.response}</Markdown>
                  </div>
                </div>
              )}
            </div>
          ))}
          {matching.length === 0 && (
            <p className="detail-empty">No entries found for this date.</p>
          )}
        </div>
      </div>
    </div>
  )
}
