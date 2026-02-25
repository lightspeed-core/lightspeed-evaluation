import { useMemo } from 'react'

export default function StatsCards({ entries }) {
  const stats = useMemo(() => {
    const dates = new Set(entries.map(e => e.date))
    const total = entries.length
    const pass = entries.filter(e => e.result === 'PASS').length
    const fail = entries.filter(e => e.result === 'FAIL').length
    const err = entries.filter(e => e.result === 'ERROR').length
    const scored = entries.filter(e => e.score !== null)
    const avg = scored.length
      ? scored.reduce((s, e) => s + e.score, 0) / scored.length
      : null
    return { runs: dates.size, total, pass, fail, err, avg }
  }, [entries])

  return (
    <div className="stats">
      <Card label="Eval Runs" value={stats.runs} cls="blue" />
      <Card label="Total Checks" value={stats.total} cls="" />
      <Card
        label="Pass Rate"
        value={stats.total ? `${(stats.pass / stats.total * 100).toFixed(1)}%` : '-'}
        cls="green"
      />
      <Card
        label="Fail Rate"
        value={stats.total ? `${(stats.fail / stats.total * 100).toFixed(1)}%` : '-'}
        cls="red"
      />
      <Card
        label="Error Rate"
        value={stats.total ? `${(stats.err / stats.total * 100).toFixed(1)}%` : '-'}
        cls="yellow"
      />
      <Card
        label="Avg Score"
        value={stats.avg !== null ? stats.avg.toFixed(3) : '-'}
        cls="blue"
      />
    </div>
  )
}

function Card({ label, value, cls }) {
  return (
    <div className="stat-card">
      <div className="label">{label}</div>
      <div className={`value ${cls}`}>{value}</div>
    </div>
  )
}
