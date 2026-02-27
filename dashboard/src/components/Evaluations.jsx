import { useState, useEffect, useMemo } from 'react'
import Papa from 'papaparse'
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import yaml from 'react-syntax-highlighter/dist/esm/languages/hljs/yaml'
import { atomOneDark, atomOneLight } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import { useTheme } from '../hooks/useTheme'
import { COLORS } from './ChartSetup'
import Markdown from 'react-markdown'
import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'

SyntaxHighlighter.registerLanguage('yaml', yaml)

const PAGE_SIZE = 20

function formatFilename(filename) {
  const match = filename.match(/evaluation_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_detailed\.csv/)
  if (!match) return filename
  const [, y, mo, d, h, mi, s] = match
  const date = new Date(`${y}-${mo}-${d}T${h}:${mi}:${s}`)
  return `${date.toLocaleDateString('en-US', {
    year: 'numeric', month: 'short', day: 'numeric',
  })} at ${date.toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
  })}`
}

function compareValues(a, b) {
  if (a == null && b == null) return 0
  if (a == null) return 1
  if (b == null) return -1
  const numA = Number(a)
  const numB = Number(b)
  if (!isNaN(numA) && !isNaN(numB)) return numA - numB
  return String(a).localeCompare(String(b))
}

function AmendedModal({ content, onClose }) {
  const { theme } = useTheme()
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content amended-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Conversation Configuration</h2>
            <div className="modal-subtitle">{content.name}</div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body">
          <SyntaxHighlighter
            language="yaml"
            style={theme === 'dark' ? atomOneDark : atomOneLight}
            customStyle={{
              margin: 0,
              borderRadius: '6px',
              fontSize: '13px',
              lineHeight: '1.6',
            }}
            showLineNumbers
          >
            {content.text}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  )
}

function AmendedLoadingOverlay() {
  return (
    <div className="modal-overlay">
      <div className="explorer-loading">
        <div className="spinner" />
        <p>Loading conversation config...</p>
      </div>
    </div>
  )
}


function GraphsModal({ file, graphs, onClose, onPreview }) {
  const title = formatFilename(file)

  const labelFromFilename = (f) => {
    const m = f.match(/evaluation_\d{8}_\d{6}_(.+)\.png$/)
    if (!m) return f
    return m[1].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content graphs-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Graphs</h2>
            <div className="modal-subtitle">{title}</div>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body">
          <div className="graphs-grid">
            {graphs.map(g => (
              <div key={g} className="graph-thumb" onClick={() => onPreview(g)}>
                <img src={`/results/graphs/${g}`} alt={labelFromFilename(g)} />
                <div className="graph-thumb-label">{labelFromFilename(g)}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

function GraphPreviewModal({ filename, onClose }) {
  const labelFromFilename = (f) => {
    const m = f.match(/evaluation_\d{8}_\d{6}_(.+)\.png$/)
    if (!m) return f
    return m[1].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
  }

  return (
    <div className="modal-overlay graph-preview-overlay" onClick={onClose}>
      <div className="graph-preview-container" onClick={e => e.stopPropagation()}>
        <button className="modal-close graph-preview-close" onClick={onClose} aria-label="Close">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
        <img src={`/results/graphs/${filename}`} alt={labelFromFilename(filename)} className="graph-preview-img" />
        <div className="graph-preview-label">{labelFromFilename(filename)}</div>
      </div>
    </div>
  )
}

const COMPARE_COLUMNS = [
  { key: 'conversation_group_id', label: 'Conversation' },
  { key: 'turn_id', label: 'Turn' },
  { key: 'metric_identifier', label: 'Metric' },
  { key: 'score_a', label: 'Score (A)' },
  { key: 'score_b', label: 'Score (B)' },
  { key: 'delta', label: 'Delta' },
  { key: 'result_a', label: 'Result (A)' },
  { key: 'result_b', label: 'Result (B)' },
  { key: 'exec_time_a', label: 'Time (A)' },
  { key: 'exec_time_b', label: 'Time (B)' },
]

const COMPARE_PAGE_SIZE = 25

function CompareModal({ data, onClose }) {
  const [sortCol, setSortCol] = useState(null)
  const [sortDir, setSortDir] = useState('asc')
  const [page, setPage] = useState(0)
  const [deltaFilter, setDeltaFilter] = useState(null)

  const titleA = formatFilename(data.fileA)
  const titleB = formatFilename(data.fileB)

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortCol(col); setSortDir('asc') }
    setPage(0)
  }

  const filtered = useMemo(() => {
    if (!deltaFilter) return data.rows
    return data.rows.filter(r => {
      if (deltaFilter === 'improved') return r.delta != null && r.delta > 0
      if (deltaFilter === 'regressed') return r.delta != null && r.delta < 0
      if (deltaFilter === 'unchanged') return r.delta != null && r.delta === 0
      return true
    })
  }, [data.rows, deltaFilter])

  const sorted = useMemo(() => {
    const rows = [...filtered]
    if (sortCol) {
      rows.sort((a, b) => {
        const cmp = compareValues(a[sortCol], b[sortCol])
        return sortDir === 'asc' ? cmp : -cmp
      })
    }
    return rows
  }, [filtered, sortCol, sortDir])

  const totalPages = Math.max(1, Math.ceil(sorted.length / COMPARE_PAGE_SIZE))
  const pageRows = sorted.slice(page * COMPARE_PAGE_SIZE, (page + 1) * COMPARE_PAGE_SIZE)

  const improved = data.rows.filter(r => r.delta != null && r.delta > 0).length
  const regressed = data.rows.filter(r => r.delta != null && r.delta < 0).length
  const unchanged = data.rows.filter(r => r.delta != null && r.delta === 0).length

  const downloadPdf = () => {
    const doc = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' })
    const pageW = doc.internal.pageSize.getWidth()
    const margin = 14

    doc.setFontSize(18)
    doc.setFont('helvetica', 'bold')
    doc.text('Evaluation Comparison', margin, 18)

    doc.setFontSize(11)
    doc.setFont('helvetica', 'normal')
    doc.text(`A: ${titleA}`, margin, 26)
    doc.text(`B: ${titleB}`, margin, 32)

    doc.setDrawColor(200)
    doc.line(margin, 35, pageW - margin, 35)

    let y = 42
    doc.setFontSize(10)
    doc.setFont('helvetica', 'bold')
    doc.setTextColor(34, 139, 34)
    doc.text(`${improved} improved`, margin, y)
    const w1 = doc.getTextWidth(`${improved} improved`) + 8
    doc.setTextColor(220, 38, 38)
    doc.text(`${regressed} regressed`, margin + w1, y)
    const w2 = w1 + doc.getTextWidth(`${regressed} regressed`) + 8
    doc.setTextColor(202, 138, 4)
    doc.text(`${unchanged} unchanged`, margin + w2, y)
    const w3 = w2 + doc.getTextWidth(`${unchanged} unchanged`) + 8
    doc.setTextColor(59, 130, 246)
    doc.text(`${data.rows.length} total rows`, margin + w3, y)
    doc.setTextColor(0)
    y += 8

    const tableColumns = COMPARE_COLUMNS.map(c => c.label)
    autoTable(doc, {
      startY: y,
      head: [tableColumns],
      body: data.rows.map(row => {
        const delta = row.delta != null ? (row.delta > 0 ? '+' : '') + row.delta.toFixed(4) : '-'
        return [
          row.conversation_group_id, row.turn_id, row.metric_identifier,
          row.score_a || '-', row.score_b || '-', delta,
          row.result_a || '-', row.result_b || '-',
          row.exec_time_a || '-', row.exec_time_b || '-',
        ]
      }),
      styles: { fontSize: 7, cellPadding: 2 },
      headStyles: { fillColor: [59, 130, 246], textColor: 255, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [245, 247, 250] },
      margin: { left: margin, right: margin },
      didParseCell: (cellData) => {
        if (cellData.section !== 'body') return
        const colIdx = cellData.column.index
        const val = cellData.cell.raw || ''
        if (colIdx === 5) {
          if (val.startsWith('+')) cellData.cell.styles.textColor = [34, 139, 34]
          else if (val.startsWith('-') && val !== '-') cellData.cell.styles.textColor = [220, 38, 38]
          cellData.cell.styles.fontStyle = 'bold'
        }
        if (colIdx === 6 || colIdx === 7) {
          const upper = String(val).toUpperCase()
          if (upper === 'PASS') cellData.cell.styles.textColor = [34, 139, 34]
          else if (upper === 'FAIL') cellData.cell.styles.textColor = [220, 38, 38]
          else if (upper === 'ERROR') cellData.cell.styles.textColor = [202, 138, 4]
        }
      },
    })

    doc.save(`comparison_${extractTimestamp(data.fileA) || 'a'}_vs_${extractTimestamp(data.fileB) || 'b'}.pdf`)
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content amended-modal compare-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Evaluation Comparison</h2>
            <div className="modal-subtitle">
              A: <strong>{titleA}</strong> &nbsp;vs&nbsp; B: <strong>{titleB}</strong>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <button className="modal-close" onClick={downloadPdf} aria-label="Download" title="Download as PDF">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            </button>
            <button className="modal-close" onClick={onClose} aria-label="Close">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        </div>
        <div className="compare-summary">
          <span className={`compare-stat improved${deltaFilter === 'improved' ? ' active' : ''}`} onClick={() => { setDeltaFilter(f => f === 'improved' ? null : 'improved'); setPage(0) }}>{improved} improved</span>
          <span className={`compare-stat regressed${deltaFilter === 'regressed' ? ' active' : ''}`} onClick={() => { setDeltaFilter(f => f === 'regressed' ? null : 'regressed'); setPage(0) }}>{regressed} regressed</span>
          <span className={`compare-stat unchanged${deltaFilter === 'unchanged' ? ' active' : ''}`} onClick={() => { setDeltaFilter(f => f === 'unchanged' ? null : 'unchanged'); setPage(0) }}>{unchanged} unchanged</span>
          <span className={`compare-stat total${deltaFilter === 'total' ? ' active' : ''}`} onClick={() => { setDeltaFilter(f => f === 'total' ? null : 'total'); setPage(0) }}>{data.rows.length} total rows</span>
        </div>
        <div className="modal-body">
          <div style={{ overflowX: 'auto' }}>
            <table className="compare-table">
              <thead>
                <tr>
                  {COMPARE_COLUMNS.map(col => (
                    <th key={col.key} className="sortable-th" onClick={() => toggleSort(col.key)}>
                      {col.label}{' '}
                      <span className="sort-indicator">
                        {sortCol === col.key ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : '\u25BD'}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row, i) => (
                  <tr key={i} className={row.only_in ? 'compare-row-missing' : ''}>
                    <td>{row.conversation_group_id}</td>
                    <td>{row.turn_id}</td>
                    <td>{row.metric_identifier}</td>
                    <td>{row.score_a}</td>
                    <td>{row.score_b}</td>
                    <td className={
                      row.delta == null ? '' :
                      row.delta > 0 ? 'compare-delta-positive' :
                      row.delta < 0 ? 'compare-delta-negative' :
                      'compare-delta-zero'
                    }>
                      {row.delta != null ? (row.delta > 0 ? '+' : '') + row.delta.toFixed(4) : '-'}
                    </td>
                    <td><span className={`badge ${(row.result_a || '').toLowerCase()}`}>{row.result_a || '-'}</span></td>
                    <td><span className={`badge ${(row.result_b || '').toLowerCase()}`}>{row.result_b || '-'}</span></td>
                    <td>{row.exec_time_a || '-'}</td>
                    <td>{row.exec_time_b || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="pager">
              <button onClick={() => setPage(0)} disabled={page === 0}>&#171;</button>
              <button onClick={() => setPage(p => p - 1)} disabled={page === 0}>&#8249;</button>
              <span className="pager-info">Page {page + 1} of {totalPages}</span>
              <button onClick={() => setPage(p => p + 1)} disabled={page >= totalPages - 1}>&#8250;</button>
              <button onClick={() => setPage(totalPages - 1)} disabled={page >= totalPages - 1}>&#187;</button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}


function DeleteConfirmModal({ filename, onConfirm, onCancel, deleting }) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-content" style={{ maxWidth: 460 }} onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Delete Evaluation</h2>
          <button className="modal-close" onClick={onCancel} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body" style={{ padding: '20px 24px' }}>
          <p style={{ marginBottom: 10 }}>Are you sure you want to permanently delete this evaluation and all related files (YAML config, graphs)?</p>
          <p style={{ fontFamily: 'monospace', fontSize: 12, color: 'var(--text2)', wordBreak: 'break-all', marginBottom: 12 }}>{filename}</p>
          <p style={{ color: 'var(--red)', fontSize: 13, fontWeight: 600 }}>This action cannot be undone.</p>
        </div>
        <div className="sc-footer">
          <button className="sc-btn sc-btn-ghost" onClick={onCancel} disabled={deleting}>Cancel</button>
          <button className="sc-btn" style={{ background: 'var(--red)', color: '#fff', borderColor: 'var(--red)' }} onClick={onConfirm} disabled={deleting}>
            {deleting ? <><span className="spinner-sm" /> Deleting...</> : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  )
}

function extractTimestamp(filename) {
  const match = filename.match(/(\d{8}_\d{6})/)
  return match ? match[1] : null
}

function fileToDate(filename) {
  const match = filename.match(/(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/)
  if (!match) return null
  const [, y, mo, d] = match
  return `${y}-${mo}-${d}`
}

const LIST_PAGE_SIZE = 10

export default function Evaluations({ reportsDir, view = 'evaluations' }) {
  const [files, setFiles] = useState([])
  const [amendedMap, setAmendedMap] = useState({})
  const [loading, setLoading] = useState(true)
  const [selectedFile, setSelectedFile] = useState(null)
  const [csvData, setCsvData] = useState(null)
  const [csvLoading, setCsvLoading] = useState(false)
  const [page, setPage] = useState(0)
  const [sortCol, setSortCol] = useState(null)
  const [sortDir, setSortDir] = useState('asc')
  const [amendedContent, setAmendedContent] = useState(null)
  const [amendedLoading, setAmendedLoading] = useState(false)
  const [expandedRows, setExpandedRows] = useState(new Set())
  const [listPage, setListPage] = useState(0)
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')
  const [evalDataGroups, setEvalDataGroups] = useState([])
  const [graphMap, setGraphMap] = useState({})
  const [graphsFile, setGraphsFile] = useState(null)
  const [pdfExporting, setPdfExporting] = useState(null)
  const [graphPreview, setGraphPreview] = useState(null)
  const [fileStats, setFileStats] = useState({})
  const [compareMode, setCompareMode] = useState(false)
  const [compareSelection, setCompareSelection] = useState([])
  const [compareData, setCompareData] = useState(null)
  const [compareLoading, setCompareLoading] = useState(false)
  const [conversationFilter, setConversationFilter] = useState('')
  const [viewerConversationFilter, setViewerConversationFilter] = useState('')
  const [deleteConfirm, setDeleteConfirm] = useState(null)
  const [deleting, setDeleting] = useState(false)
  const [summaryMap, setSummaryMap] = useState({})


  useEffect(() => {
    Promise.all([
      fetch('/api/manifest').then(r => r.json()),
      fetch('/api/amended-files').then(r => r.json()),
      fetch('/api/eval-data').then(r => r.json()),
      fetch('/api/eval-graphs').then(r => r.json()),
      fetch('/api/eval-summaries').then(r => r.json()).catch(() => ({})),
    ])
      .then(([manifest, amendedFiles, evalData, graphs, summaries]) => {
        const csvFiles = manifest.files || manifest
        setFiles(csvFiles.sort().reverse())
        const map = {}
        for (const af of amendedFiles) {
          const ts = extractTimestamp(af)
          if (ts) map[ts] = af
        }
        setAmendedMap(map)
        setEvalDataGroups(evalData.groups || [])
        setGraphMap(graphs)
        setSummaryMap(summaries)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  useEffect(() => {
    if (!files.length) return
    const stats = {}
    Promise.all(files.map(file =>
      fetch(`/results/${file}`)
        .then(r => r.text())
        .then(text => {
          Papa.parse(text, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
              const rows = results.data
              const passed = rows.filter(r => (r.result || '').toUpperCase() === 'PASS').length
              const conversations = [...new Set(rows.map(r => r.conversation_group_id).filter(Boolean))]
              stats[file] = { passed, total: rows.length, conversations }
            },
          })
        })
        .catch(() => {})
    )).then(() => setFileStats(stats))
  }, [files])


  const buildCompareRow = (a, b) => {
    const scoreA = a ? parseFloat(a.score) : null
    const scoreB = b ? parseFloat(b.score) : null
    const delta = (!isNaN(scoreA) && !isNaN(scoreB) && scoreA != null && scoreB != null)
      ? scoreB - scoreA : null
    return {
      conversation_group_id: a?.conversation_group_id || b?.conversation_group_id,
      turn_id: a?.turn_id || b?.turn_id,
      metric_identifier: a?.metric_identifier || b?.metric_identifier,
      score_a: a?.score ?? '',
      score_b: b?.score ?? '',
      delta,
      result_a: a?.result ?? '',
      result_b: b?.result ?? '',
      exec_time_a: a?.execution_time ?? '',
      exec_time_b: b?.execution_time ?? '',
      only_in: a && b ? null : (a ? 'A' : 'B'),
    }
  }

  const handleCompare = async () => {
    if (compareSelection.length !== 2) return
    setCompareLoading(true)
    try {
      const sorted = [...compareSelection].sort((a, b) =>
        (extractTimestamp(a) || '').localeCompare(extractTimestamp(b) || '')
      )
      const [textA, textB] = await Promise.all(
        sorted.map(f => fetch(`/results/${f}`).then(r => r.text()))
      )
      const parseCSV = (text) => Papa.parse(text, { header: true, skipEmptyLines: true }).data
      const rowsA = parseCSV(textA)
      const rowsB = parseCSV(textB)

      const keyOf = (row) =>
        `${row.conversation_group_id}||${row.turn_id}||${row.metric_identifier}`

      const mapB = new Map()
      for (const row of rowsB) mapB.set(keyOf(row), row)

      const seen = new Set()
      const comparedRows = []

      for (const rowA of rowsA) {
        const key = keyOf(rowA)
        seen.add(key)
        comparedRows.push(buildCompareRow(rowA, mapB.get(key) || null))
      }

      for (const rowB of rowsB) {
        if (!seen.has(keyOf(rowB))) {
          comparedRows.push(buildCompareRow(null, rowB))
        }
      }

      setCompareData({
        fileA: sorted[0],
        fileB: sorted[1],
        rows: comparedRows,
      })
      setCompareMode(false)
      setCompareSelection([])
    } catch { /* ignore */ }
    setCompareLoading(false)
  }

  const openFile = async (filename) => {
    setSelectedFile(filename)
    setCsvLoading(true)
    try {
      const res = await fetch(`/results/${filename}`)
      const text = await res.text()
      Papa.parse(text, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          setCsvData(results)
          setCsvLoading(false)
        },
        error: () => setCsvLoading(false),
      })
    } catch {
      setCsvLoading(false)
    }
  }

  const closeViewer = () => {
    setSelectedFile(null)
    setCsvData(null)
    setPage(0)
    setSortCol(null)
    setSortDir('asc')
    setAmendedContent(null)
    setViewerConversationFilter('')
  }

  const openAmended = async (filename, e) => {
    e.stopPropagation()
    const ts = extractTimestamp(filename)
    const amendedFile = ts && amendedMap[ts]
    if (!amendedFile) return
    setAmendedLoading(true)
    try {
      const res = await fetch(`/results/${amendedFile}`)
      const text = await res.text()
      setAmendedContent({ name: amendedFile, text })
    } catch {
      setAmendedContent(null)
    }
    setAmendedLoading(false)
  }

  const closeAmended = () => setAmendedContent(null)

  const exportPdf = async (filename, e) => {
    e.stopPropagation()
    if (pdfExporting) return
    setPdfExporting(filename)
    // Yield to let UI update before heavy work
    await new Promise(r => setTimeout(r, 50))
    try {
    const title = formatFilename(filename)
    const ts = extractTimestamp(filename)

    // Fetch CSV
    const csvRes = await fetch(`/results/${filename}`)
    const csvText = await csvRes.text()
    const parsed = Papa.parse(csvText, { header: true, skipEmptyLines: true })
    const columns = parsed.meta?.fields || []
    const rows = parsed.data

    // Fetch amended YAML if available
    const amendedFile = ts && amendedMap[ts]
    let yamlText = null
    if (amendedFile) {
      const yamlRes = await fetch(`/results/${amendedFile}`)
      yamlText = await yamlRes.text()
    }

    // Build PDF (landscape for wide tables)
    const doc = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' })
    const pageW = doc.internal.pageSize.getWidth()
    const margin = 14

    // Title
    doc.setFontSize(18)
    doc.setFont('helvetica', 'bold')
    doc.text('Lightspeed Evaluation Report', margin, 18)
    doc.setFontSize(12)
    doc.setFont('helvetica', 'normal')
    doc.text(title, margin, 26)
    doc.setDrawColor(200)
    doc.line(margin, 29, pageW - margin, 29)

    // Summary stats
    let y = 36
    const pass = rows.filter(r => (r.result || '').toUpperCase() === 'PASS').length
    const fail = rows.filter(r => (r.result || '').toUpperCase() === 'FAIL').length
    const error = rows.filter(r => (r.result || '').toUpperCase() === 'ERROR').length
    const scores = rows.map(r => parseFloat(r.score)).filter(n => !isNaN(n))
    const avgScore = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length) : null

    doc.setFontSize(11)
    doc.setFont('helvetica', 'bold')
    doc.text('Summary', margin, y)
    y += 6
    doc.setFont('helvetica', 'normal')
    doc.setFontSize(10)
    doc.text(`Total entries: ${rows.length}    Pass: ${pass}    Fail: ${fail}    Error: ${error}    Avg Score: ${avgScore !== null ? avgScore.toFixed(4) : 'N/A'}`, margin, y)
    y += 10

    // Results table
    const tableColumns = columns.filter(c => c !== 'response')
    doc.setFontSize(11)
    doc.setFont('helvetica', 'bold')
    doc.text('Evaluation Results', margin, y)
    y += 2

    autoTable(doc, {
      startY: y,
      head: [tableColumns],
      body: rows.map(r => tableColumns.map(c => {
        const val = r[c] || ''
        return val.length > 120 ? val.slice(0, 117) + '...' : val
      })),
      styles: { fontSize: 7, cellPadding: 2 },
      headStyles: { fillColor: [59, 130, 246], textColor: 255, fontStyle: 'bold' },
      alternateRowStyles: { fillColor: [245, 247, 250] },
      margin: { left: margin, right: margin },
      didParseCell: (data) => {
        if (data.section === 'body' && tableColumns[data.column.index] === 'result') {
          const val = (data.cell.raw || '').toUpperCase()
          if (val === 'PASS') data.cell.styles.textColor = [34, 139, 34]
          else if (val === 'FAIL') data.cell.styles.textColor = [220, 38, 38]
          else if (val === 'ERROR') data.cell.styles.textColor = [202, 138, 4]
        }
      },
    })

    // Graphs page
    const graphFiles = ts && graphMap[ts]
    if (graphFiles && graphFiles.length > 0) {
      const labelFromFilename = (f) => {
        const m = f.match(/evaluation_\d{8}_\d{6}_(.+)\.png$/)
        if (!m) return f
        return m[1].replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
      }

      const loadImage = (url) => new Promise((resolve) => {
        const img = new Image()
        img.crossOrigin = 'anonymous'
        img.onload = () => {
          const canvas = document.createElement('canvas')
          canvas.width = img.naturalWidth
          canvas.height = img.naturalHeight
          canvas.getContext('2d').drawImage(img, 0, 0)
          resolve({ data: canvas.toDataURL('image/png'), w: img.naturalWidth, h: img.naturalHeight })
        }
        img.onerror = () => resolve(null)
        img.src = url
      })

      const images = (await Promise.all(
        graphFiles.map(gf => loadImage(`/results/graphs/${gf}`).then(img => img ? { ...img, label: labelFromFilename(gf) } : null))
      )).filter(Boolean)

      if (images.length > 0) {
        doc.addPage()
        const pageH = doc.internal.pageSize.getHeight()
        doc.setFontSize(14)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(0)
        doc.text('Graphs', margin, 18)
        doc.setDrawColor(200)
        doc.line(margin, 21, pageW - margin, 21)

        let gy = 26
        const maxImgW = (pageW - margin * 2 - 10) / 2

        for (let i = 0; i < images.length; i += 2) {
          const pair = images.slice(i, i + 2)
          let rowH = 0

          for (let j = 0; j < pair.length; j++) {
            const img = pair[j]
            const ratio = img.h / img.w
            const imgW = maxImgW
            const imgH = imgW * ratio
            rowH = Math.max(rowH, imgH + 10)
          }

          if (gy + rowH > pageH - 10) {
            doc.addPage()
            gy = 14
          }

          for (let j = 0; j < pair.length; j++) {
            const img = pair[j]
            const ratio = img.h / img.w
            const imgW = maxImgW
            const imgH = imgW * ratio
            const x = margin + j * (maxImgW + 10)

            doc.addImage(img.data, 'PNG', x, gy, imgW, imgH)
            doc.setFontSize(8)
            doc.setFont('helvetica', 'normal')
            doc.setTextColor(100)
            doc.text(img.label, x + imgW / 2, gy + imgH + 4, { align: 'center' })
          }

          gy += rowH + 6
        }
        doc.setTextColor(0)
      }
    }

    // Conversation config (YAML) on new page as code block
    if (yamlText) {
      doc.addPage()
      const pageH = doc.internal.pageSize.getHeight()
      doc.setFontSize(14)
      doc.setFont('helvetica', 'bold')
      doc.text('Conversation Configuration', margin, 18)
      doc.setFontSize(9)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(100)
      doc.text(amendedFile, margin, 24)
      doc.setTextColor(0)
      doc.setDrawColor(200)
      doc.line(margin, 27, pageW - margin, 27)

      const codePad = 4
      const codeMargin = margin + 2
      const lineH = 3.5
      const codeW = pageW - margin * 2
      const lines = doc.splitTextToSize(yamlText, codeW - codePad * 2)

      let ly = 33
      const startCodeBlock = () => {
        const blockH = Math.min(
          lines.length * lineH + codePad * 2,
          pageH - ly - 10
        )
        doc.setFillColor(245, 247, 250)
        doc.setDrawColor(210, 215, 220)
        doc.roundedRect(margin, ly - codePad, codeW, blockH, 2, 2, 'FD')
      }

      startCodeBlock()
      ly += codePad - 1
      doc.setFontSize(7.5)
      doc.setFont('courier', 'normal')

      for (let i = 0; i < lines.length; i++) {
        if (ly + lineH > pageH - 10) {
          doc.addPage()
          ly = 14
          const remainH = Math.min((lines.length - i) * lineH + codePad * 2, pageH - ly - 10)
          doc.setFillColor(245, 247, 250)
          doc.setDrawColor(210, 215, 220)
          doc.roundedRect(margin, ly - codePad, codeW, remainH, 2, 2, 'FD')
          ly += codePad - 1
          doc.setFont('courier', 'normal')
          doc.setFontSize(7.5)
        }
        const line = lines[i]
        const keyMatch = line.match(/^(\s*)([\w_-]+)(:)(.*)/)
        if (keyMatch) {
          const [, indent, key, colon, rest] = keyMatch
          const x = codeMargin
          doc.setTextColor(0, 90, 170)
          doc.text(indent + key, x, ly)
          const keyW = doc.getTextWidth(indent + key)
          doc.setTextColor(60)
          doc.text(colon + rest, x + keyW, ly)
        } else {
          doc.setTextColor(60)
          doc.text(line, codeMargin, ly)
        }
        ly += lineH
      }
      doc.setTextColor(0)
    }

    const safeName = filename.replace(/_detailed\.csv$/, '')
    doc.save(`${safeName}_report.pdf`)
    } finally {
      setPdfExporting(null)
    }
  }

  const toggleSort = (col) => {
    if (sortCol === col) {
      setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    } else {
      setSortCol(col)
      setSortDir('asc')
    }
    setPage(0)
  }

  const handleDelete = async () => {
    if (!deleteConfirm) return
    setDeleting(true)
    try {
      const res = await fetch('/api/delete-evaluation', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: deleteConfirm }),
      })
      if (res.ok) {
        const ts = extractTimestamp(deleteConfirm)
        setFiles(prev => prev.filter(f => f !== deleteConfirm))
        setFileStats(prev => { const next = { ...prev }; delete next[deleteConfirm]; return next })
        if (ts) {
          setAmendedMap(prev => { const next = { ...prev }; delete next[ts]; return next })
          setGraphMap(prev => { const next = { ...prev }; delete next[ts]; return next })
          setSummaryMap(prev => { const next = { ...prev }; delete next[ts]; return next })
        }
      }
    } catch { /* ignore */ }
    setDeleting(false)
    setDeleteConfirm(null)
  }

  const allConversations = useMemo(() => {
    const set = new Set()
    for (const stat of Object.values(fileStats)) {
      for (const s of (stat.conversations || [])) set.add(s)
    }
    return [...set].sort()
  }, [fileStats])

  const conversationColorMap = useMemo(() => {
    const map = {}
    allConversations.forEach((s, i) => { map[s] = COLORS[i % COLORS.length] })
    return map
  }, [allConversations])

  const filteredFiles = useMemo(() => {
    return files.filter(f => {
      const d = fileToDate(f)
      if (!d) return true
      if (dateFrom && d < dateFrom) return false
      if (dateTo && d > dateTo) return false
      if (conversationFilter && fileStats[f]) {
        if (!(fileStats[f].conversations || []).includes(conversationFilter)) return false
      }
      return true
    })
  }, [files, dateFrom, dateTo, conversationFilter, fileStats])

  const compatibleFiles = useMemo(() => {
    if (!compareMode || compareSelection.length !== 1) return null
    const firstConversations = new Set(fileStats[compareSelection[0]]?.conversations || [])
    if (firstConversations.size === 0) return null // stats not loaded yet, allow all
    const result = new Set()
    for (const f of filteredFiles) {
      if ((fileStats[f]?.conversations || []).some(s => firstConversations.has(s))) {
        result.add(f)
      }
    }
    return result
  }, [compareMode, compareSelection, fileStats, filteredFiles])

  const viewerConversations = useMemo(() => {
    if (!csvData) return []
    return [...new Set(csvData.data.map(r => r.conversation_group_id).filter(Boolean))].sort()
  }, [csvData])

  const sortedRows = useMemo(() => {
    if (!csvData) return []
    let rows = [...csvData.data]
    if (viewerConversationFilter) {
      rows = rows.filter(r => r.conversation_group_id === viewerConversationFilter)
    }
    if (sortCol) {
      rows.sort((a, b) => {
        const cmp = compareValues(a[sortCol], b[sortCol])
        return sortDir === 'asc' ? cmp : -cmp
      })
    }
    return rows
  }, [csvData, sortCol, sortDir, viewerConversationFilter])

  const tagGroups = useMemo(() => {
    const grouped = {}
    for (const g of evalDataGroups) {
      const t = g.tag || 'eval'
      if (!grouped[t]) grouped[t] = []
      grouped[t].push(g)
    }
    return Object.entries(grouped).sort(([a], [b]) => a.localeCompare(b))
  }, [evalDataGroups])

  if (loading) {
    return (
      <div className="explorer-loading">
        <div className="spinner" />
        <p>Loading file list...</p>
      </div>
    )
  }

  if (selectedFile && csvData) {
    const columns = csvData.meta?.fields || []
    const totalRows = sortedRows.length
    const totalPages = Math.max(1, Math.ceil(totalRows / PAGE_SIZE))
    const start = page * PAGE_SIZE
    const pageRows = sortedRows.slice(start, start + PAGE_SIZE)

    return (
      <div className="explorer-viewer">
        <div className="viewer-header">
          <button className="viewer-back" onClick={closeViewer}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="15 18 9 12 15 6" />
            </svg>
            Back to file list
          </button>
          <h3>{formatFilename(selectedFile)}</h3>
          <span className="viewer-meta">
            {totalRows} rows &middot; {columns.length} columns
            {summaryMap[extractTimestamp(selectedFile)]?.model && (
              <span className="model-tag">{summaryMap[extractTimestamp(selectedFile)].model}</span>
            )}
          </span>
          {viewerConversations.length > 1 && (
            <div className="date-field" style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
              <label>Conversation</label>
              <select
                value={viewerConversationFilter}
                onChange={e => { setViewerConversationFilter(e.target.value); setPage(0) }}
              >
                <option value="">All</option>
                {viewerConversations.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            </div>
          )}
          {amendedMap[extractTimestamp(selectedFile)] && (
            <button className="amended-btn" onClick={(e) => openAmended(selectedFile, e)} title="View conversation configuration">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
              </svg>
              Conversation Config
            </button>
          )}
          {graphMap[extractTimestamp(selectedFile)] && (
            <button className="graphs-btn" onClick={(e) => { e.stopPropagation(); setGraphsFile(selectedFile) }} title="View graphs">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="20" x2="18" y2="10" />
                <line x1="12" y1="20" x2="12" y2="4" />
                <line x1="6" y1="20" x2="6" y2="14" />
              </svg>
              Graphs
            </button>
          )}
          <button className="export-pdf-btn" onClick={(e) => exportPdf(selectedFile, e)} disabled={!!pdfExporting}>
            {pdfExporting === selectedFile ? (
              <><div className="spinner-sm" /> Exporting...</>
            ) : (
              <><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg> Export PDF</>
            )}
          </button>
        </div>
        <div className="table-card">
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  {columns.map(col => (
                    <th key={col} className="sortable-th" onClick={() => toggleSort(col)}>
                      {col}{' '}
                      <span className="sort-indicator">
                        {sortCol === col ? (sortDir === 'asc' ? '\u25B2' : '\u25BC') : '\u25BD'}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row, i) => {
                  const rowKey = start + i
                  const isRowExpanded = expandedRows.has(rowKey)
                  return (
                    <tr key={rowKey} className={`expandable-row${isRowExpanded ? ' expanded' : ''}`} onClick={() => {
                      setExpandedRows(prev => {
                        const next = new Set(prev)
                        next.has(rowKey) ? next.delete(rowKey) : next.add(rowKey)
                        return next
                      })
                    }}>
                      <td className="row-num">{rowKey + 1}</td>
                      {columns.map(col => {
                        const val = row[col] || ''
                        const isMd = (col === 'reason' || col === 'response') && val
                        const isLong = col === 'query'
                        return (
                          <td
                            key={col}
                            className={isMd ? `explorer-cell-md${isRowExpanded ? ' expanded' : ''}` : isLong ? `explorer-cell-long${isRowExpanded ? ' expanded' : ''}` : ''}
                          >
                            {col === 'result' ? (
                              <span className={`badge ${val.toLowerCase()}`}>{val}</span>
                            ) : isMd ? (
                              isRowExpanded
                                ? <div className="md-expanded"><div className="detail-md"><Markdown>{val}</Markdown></div></div>
                                : <div className="md-collapsed">{val}</div>
                            ) : (
                              val
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {totalPages > 1 && (
            <div className="pager">
              <button onClick={() => setPage(0)} disabled={page === 0}>&#171;</button>
              <button onClick={() => setPage(p => p - 1)} disabled={page === 0}>&#8249;</button>
              <span className="pager-info">Page {page + 1} of {totalPages}</span>
              <button onClick={() => setPage(p => p + 1)} disabled={page >= totalPages - 1}>&#8250;</button>
              <button onClick={() => setPage(totalPages - 1)} disabled={page >= totalPages - 1}>&#187;</button>
            </div>
          )}
        </div>
        {graphsFile && graphMap[extractTimestamp(graphsFile)] && (
          <GraphsModal
            file={graphsFile}
            graphs={graphMap[extractTimestamp(graphsFile)]}
            onClose={() => setGraphsFile(null)}
            onPreview={(g) => setGraphPreview(g)}
          />
        )}
        {graphPreview && (
          <GraphPreviewModal
            filename={graphPreview}
            onClose={() => setGraphPreview(null)}
          />
        )}
        {amendedContent && <AmendedModal content={amendedContent} onClose={closeAmended} />}
        {amendedLoading && <AmendedLoadingOverlay />}
      </div>
    )
  }

  if (selectedFile && csvLoading) {
    return (
      <div className="explorer-loading">
        <div className="spinner" />
        <p>Loading CSV...</p>
      </div>
    )
  }

  const listTotalPages = Math.max(1, Math.ceil(filteredFiles.length / LIST_PAGE_SIZE))
  const listStart = listPage * LIST_PAGE_SIZE
  const pageFiles = filteredFiles.slice(listStart, listStart + LIST_PAGE_SIZE)

  return (
    <div className="explorer-list">
      {view === 'evaluations' && reportsDir && !reportsDir.exists && (
        <div className="run-disclaimer run-disclaimer-warn" style={{ marginBottom: 16 }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <span>Reports directory <code>{reportsDir.path}</code> does not exist. Run an evaluation to create it, or check your <code>LS_EVAL_REPORTS_PATH</code> setting.</span>
        </div>
      )}

      {view === 'evaluations' && (
        <>
          <div className="explorer-list-header">
            <div className="compare-actions">
              <button
                className={`compare-toggle-btn${compareMode ? (compareSelection.length === 2 ? ' cancel' : ' active') : ''}`}
                onClick={() => { setCompareMode(m => !m); setCompareSelection([]) }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="2" y="4" width="8" height="16" rx="1" />
                  <rect x="14" y="4" width="8" height="16" rx="1" />
                </svg>
                {compareMode ? 'Cancel' : 'Compare'}
              </button>
              {compareMode && compareSelection.length === 2 && (
                <button className="compare-go-btn" onClick={handleCompare} disabled={compareLoading}>
                  {compareLoading ? (
                    <><div className="spinner-sm" /> Loading...</>
                  ) : (
                    <>Compare</>
                  )}
                </button>
              )}
            </div>
            <div className="date-filters">
              {allConversations.length > 0 && (
                <div className="date-field">
                  <label>Conversation</label>
                  <select
                    value={conversationFilter}
                    onChange={e => { setConversationFilter(e.target.value); setListPage(0) }}
                  >
                    <option value="">All conversations</option>
                    {allConversations.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
              )}
              <div className="date-field">
                <label>From</label>
                <input type="date" value={dateFrom} onChange={e => { setDateFrom(e.target.value); setListPage(0) }} />
              </div>
              <div className="date-field">
                <label>To</label>
                <input type="date" value={dateTo} onChange={e => { setDateTo(e.target.value); setListPage(0) }} />
              </div>
              {(dateFrom || dateTo || conversationFilter) && (
                <button className="date-clear" onClick={() => { setDateFrom(''); setDateTo(''); setConversationFilter(''); setListPage(0) }}>Clear</button>
              )}
            </div>
          </div>
          {filteredFiles.length === 0 ? (
            <p className="explorer-empty">No evaluation files found{(dateFrom || dateTo) ? ' for the selected date range' : ' in results/'}</p>
          ) : (
            <>
              <div className="explorer-files">
                {pageFiles.map(file => {
                  const isIncompatible = compareMode && compareSelection.length === 1
                    && !compareSelection.includes(file)
                    && compatibleFiles !== null && !compatibleFiles.has(file)
                  return (
                  <div key={file} className={`explorer-file-card${compareMode && compareSelection.includes(file) ? ' compare-selected' : ''}${isIncompatible ? ' compare-incompatible' : ''}`} onClick={() => {
                    if (compareMode) {
                      if (isIncompatible) return
                      setCompareSelection(prev => {
                        if (prev.includes(file)) return prev.filter(f => f !== file)
                        if (prev.length >= 2) return prev
                        return [...prev, file]
                      })
                    } else {
                      openFile(file)
                    }
                  }}>
                    {compareMode && (
                      <input
                        type="checkbox"
                        className="compare-checkbox"
                        checked={compareSelection.includes(file)}
                        readOnly
                      />
                    )}
                    <div className="explorer-file-icon" style={{ fontSize: '24px' }}>
                      🚀
                      {fileStats[file] && (() => {
                        const { passed, total } = fileStats[file]
                        const half = total / 2
                        const color = passed > half ? 'var(--green)' : passed === half ? 'var(--yellow)' : 'var(--red)'
                        return (
                          <span style={{ fontSize: '16px', fontWeight: 700, color, marginLeft: 8, fontFamily: 'monospace' }}>
                            {passed}/{total}
                          </span>
                        )
                      })()}
                    </div>
                    <div className="explorer-file-info">
                      <span className="explorer-file-name">{formatFilename(file)}</span>
                      <span className="explorer-file-raw">{file}</span>
                      {!conversationFilter && fileStats[file]?.conversations?.length > 0 && (
                        <div className="explorer-file-conversations">
                          {fileStats[file].conversations.map(s => {
                            const c = conversationColorMap[s] || COLORS[0]
                            return (
                              <span
                                key={s}
                                className="conversation-tag"
                                style={{
                                  color: c,
                                  borderColor: c + '66',
                                  background: c + '18',
                                }}
                              >{s}</span>
                            )
                          })}
                        </div>
                      )}
                    </div>
                    {summaryMap[extractTimestamp(file)]?.model && (
                      <span className="model-tag">{summaryMap[extractTimestamp(file)].model}</span>
                    )}
                    <button
                      className="delete-eval-btn"
                      onClick={(e) => { e.stopPropagation(); setDeleteConfirm(file) }}
                      title="Delete evaluation"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                        <path d="M10 11v6M14 11v6" />
                        <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                      </svg>
                    </button>
                  </div>
                  )
                })}
              </div>
              {listTotalPages > 1 && (
                <div className="pager">
                  <button onClick={() => setListPage(0)} disabled={listPage === 0}>&#171;</button>
                  <button onClick={() => setListPage(p => p - 1)} disabled={listPage === 0}>&#8249;</button>
                  <span className="pager-info">Page {listPage + 1} of {listTotalPages}</span>
                  <button onClick={() => setListPage(p => p + 1)} disabled={listPage >= listTotalPages - 1}>&#8250;</button>
                  <button onClick={() => setListPage(listTotalPages - 1)} disabled={listPage >= listTotalPages - 1}>&#187;</button>
                </div>
              )}
            </>
          )}
        </>
      )}

      {view === 'conversations' && (
        <>
          {tagGroups.length === 0 ? (
            <p className="explorer-empty">No conversation groups found in eval.yaml</p>
          ) : (
            <div className="explorer-files">
              {tagGroups.map(([tag, groups]) => (
                <div key={tag}>
                  <div className="explorer-file-card" style={{ borderLeft: '3px solid var(--accent)', background: 'var(--card-bg)', marginBottom: 0, borderRadius: '8px 8px 0 0' }}>
                    <div className="explorer-file-icon">
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z" />
                        <line x1="7" y1="7" x2="7.01" y2="7" />
                      </svg>
                    </div>
                    <div className="explorer-file-info">
                      <span className="explorer-file-name" style={{ fontSize: '15px' }}>{tag}</span>
                      <span className="explorer-file-raw">{groups.length} group(s), {groups.reduce((s, g) => s + g.turnCount, 0)} turn(s)</span>
                    </div>
                  </div>
                  {groups.map(g => (
                    <div key={g.conversation_group_id} className="explorer-file-card" style={{ marginLeft: 16, marginTop: 0, borderRadius: 0, borderTop: '1px solid var(--border)' }}>
                      <div className="explorer-file-icon">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                          <polyline points="14 2 14 8 20 8" />
                        </svg>
                      </div>
                      <div className="explorer-file-info">
                        <span className="explorer-file-name">{g.conversation_group_id}</span>
                        <span className="explorer-file-raw">{g.turnCount} turn(s): {g.turnIds.join(', ')}</span>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {graphsFile && graphMap[extractTimestamp(graphsFile)] && (
        <GraphsModal
          file={graphsFile}
          graphs={graphMap[extractTimestamp(graphsFile)]}
          onClose={() => setGraphsFile(null)}
          onPreview={(g) => setGraphPreview(g)}
        />
      )}
      {graphPreview && (
        <GraphPreviewModal
          filename={graphPreview}
          onClose={() => setGraphPreview(null)}
        />
      )}
      {amendedContent && <AmendedModal content={amendedContent} onClose={closeAmended} />}
      {amendedLoading && <AmendedLoadingOverlay />}
      {compareData && <CompareModal data={compareData} onClose={() => setCompareData(null)} />}
      {compareLoading && <AmendedLoadingOverlay />}
      {deleteConfirm && (
        <DeleteConfirmModal
          filename={deleteConfirm}
          onConfirm={handleDelete}
          onCancel={() => setDeleteConfirm(null)}
          deleting={deleting}
        />
      )}
    </div>
  )
}
