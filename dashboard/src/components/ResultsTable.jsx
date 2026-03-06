import { useState, useMemo } from 'react'
import Markdown from 'react-markdown'

const PAGE_SIZE = 20

const COLUMNS = [
  { key: 'date', label: 'Date', type: 'date' },
  { key: 'conversation', label: 'Conversation' },
  { key: 'turn', label: 'Turn' },
  { key: 'metric', label: 'Metric' },
  { key: 'score', label: 'Score', type: 'number' },
  { key: 'result', label: 'Result', type: 'badge' },
  { key: 'query', label: 'Query', type: 'long' },
  { key: 'reason', label: 'Reason', type: 'md' },
  { key: 'response', label: 'Response', type: 'md' },
  { key: 'time', label: 'Time (s)', type: 'number' },
]

function getValue(e, key) {
  switch (key) {
    case 'date': return e.date
    case 'conversation': return e.conversationGroupId
    case 'turn': return e.turnId
    case 'metric': return e.metric
    case 'score': return e.score
    case 'result': return e.result
    case 'query': return e.query
    case 'reason': return e.reason
    case 'response': return e.response
    case 'time': return e.executionTime
    default: return ''
  }
}

function compare(a, b) {
  if ((a === null || a === undefined) && (b === null || b === undefined)) return 0
  if (a === null || a === undefined) return 1
  if (b === null || b === undefined) return -1
  if (typeof a === 'number' && typeof b === 'number') return a - b
  return String(a).localeCompare(String(b))
}

export default function ResultsTable({ entries }) {
  const [page, setPage] = useState(0)
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')
  const [sortKey, setSortKey] = useState(null)
  const [sortDir, setSortDir] = useState('asc')
  const [expandedRows, setExpandedRows] = useState(new Set())

  const dateFiltered = useMemo(() => {
    let result = entries
    if (startDate) {
      const start = new Date(startDate + 'T00:00:00')
      result = result.filter(e => new Date(e.date) >= start)
    }
    if (endDate) {
      const end = new Date(endDate + 'T23:59:59')
      result = result.filter(e => new Date(e.date) <= end)
    }
    return result
  }, [entries, startDate, endDate])

  const sorted = useMemo(() => {
    if (!sortKey) return dateFiltered
    const mult = sortDir === 'asc' ? 1 : -1
    return [...dateFiltered].sort((a, b) => mult * compare(getValue(a, sortKey), getValue(b, sortKey)))
  }, [dateFiltered, sortKey, sortDir])

  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE))
  const safePage = Math.min(page, totalPages - 1)
  const pageEntries = sorted.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE)

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortKey(key)
      setSortDir('asc')
    }
    setPage(0)
  }

  const clearDates = () => {
    setStartDate('')
    setEndDate('')
    setPage(0)
  }

  const toggleRow = (rowKey) => {
    setExpandedRows(prev => {
      const next = new Set(prev)
      next.has(rowKey) ? next.delete(rowKey) : next.add(rowKey)
      return next
    })
  }

  const renderCell = (entry, col, isExpanded) => {
    const val = getValue(entry, col.key)

    if (col.type === 'date') {
      return <td key={col.key}>{new Date(val).toLocaleString()}</td>
    }
    if (col.type === 'number') {
      const num = Number(val)
      const formatted = Number.isFinite(num)
        ? (col.key === 'score' ? num.toFixed(4) : num.toFixed(3))
        : '-'
      return <td key={col.key}>{formatted}</td>
    }
    if (col.type === 'badge') {
      return (
        <td key={col.key}>
          <span className={`badge ${(val || '').toLowerCase()}`}>{val}</span>
        </td>
      )
    }
    if (col.type === 'long') {
      return (
        <td key={col.key} className={`explorer-cell-long${isExpanded ? ' expanded' : ''}`}>
          {val || ''}
        </td>
      )
    }
    if (col.type === 'md') {
      const text = val || ''
      if (!text) return <td key={col.key}>-</td>
      return (
        <td key={col.key} className={`explorer-cell-md${isExpanded ? ' expanded' : ''}`}>
          {isExpanded
            ? <div className="md-expanded"><div className="detail-md"><Markdown>{text}</Markdown></div></div>
            : <div className="md-collapsed">{text}</div>
          }
        </td>
      )
    }
    return <td key={col.key}>{val || ''}</td>
  }

  return (
    <div className="table-card">
      <div className="table-header">
        <h3>All Evaluation Results</h3>
        <div className="date-filters">
          <div className="date-field">
            <label>From</label>
            <input
              type="date"
              value={startDate}
              onChange={e => { setStartDate(e.target.value); setPage(0) }}
            />
          </div>
          <div className="date-field">
            <label>To</label>
            <input
              type="date"
              value={endDate}
              onChange={e => { setEndDate(e.target.value); setPage(0) }}
            />
          </div>
          {(startDate || endDate) && (
            <button className="date-clear" onClick={clearDates}>Clear</button>
          )}
          <span className="date-count">{sorted.length} results</span>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead>
            <tr>
              <th>#</th>
              {COLUMNS.map(col => (
                <th
                  key={col.key}
                  className="sortable-th"
                  onClick={() => handleSort(col.key)}
                >
                  {col.label}
                  <span className="sort-indicator">
                    {sortKey === col.key ? (sortDir === 'asc' ? ' \u25B2' : ' \u25BC') : ' \u25BD'}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageEntries.map((e, i) => {
              const displayNum = safePage * PAGE_SIZE + i + 1
              const stableId = `${e.date || ''}||${e.conversationGroupId || ''}||${e.turnId || ''}||${e.metric || ''}`
              const isExpanded = expandedRows.has(stableId)
              return (
                <tr key={stableId} className={`expandable-row${isExpanded ? ' expanded' : ''}`}>
                  <td className="row-num">
                    <button
                      className="row-toggle"
                      aria-expanded={isExpanded}
                      aria-label={`${isExpanded ? 'Collapse' : 'Expand'} row ${displayNum}`}
                      onClick={() => toggleRow(stableId)}
                    >
                      {isExpanded ? '\u25BC' : '\u25B6'} {displayNum}
                    </button>
                  </td>
                  {COLUMNS.map(col => renderCell(e, col, isExpanded))}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      <div className="pager">
        <button
          disabled={safePage === 0}
          onClick={() => setPage(0)}
        >
          &laquo;
        </button>
        <button
          disabled={safePage === 0}
          onClick={() => setPage(p => p - 1)}
        >
          &lsaquo;
        </button>
        <span className="pager-info">
          Page {safePage + 1} of {totalPages}
        </span>
        <button
          disabled={safePage >= totalPages - 1}
          onClick={() => setPage(p => p + 1)}
        >
          &rsaquo;
        </button>
        <button
          disabled={safePage >= totalPages - 1}
          onClick={() => setPage(totalPages - 1)}
        >
          &raquo;
        </button>
      </div>
    </div>
  )
}
