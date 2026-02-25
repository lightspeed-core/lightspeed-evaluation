import { useMemo } from 'react'
import { Bar } from 'react-chartjs-2'
import { useChartTheme } from '../hooks/useTheme'

export default function StackedBarChart({ entries }) {
  const ct = useChartTheme()

  const { data, options } = useMemo(() => {
    const byDate = {}
    entries.forEach(e => {
      if (!byDate[e.date]) byDate[e.date] = { pass: 0, fail: 0, error: 0 }
      if (e.result === 'PASS') byDate[e.date].pass++
      else if (e.result === 'FAIL') byDate[e.date].fail++
      else byDate[e.date].error++
    })
    const dates = Object.keys(byDate).sort()
    const labels = dates.map(d => {
      const dt = new Date(d)
      return dt.toLocaleDateString() + ' ' + dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    })
    return {
      data: {
        labels,
        datasets: [
          { label: 'PASS', data: dates.map(d => byDate[d].pass), backgroundColor: '#3fb950' },
          { label: 'FAIL', data: dates.map(d => byDate[d].fail), backgroundColor: '#f85149' },
          { label: 'ERROR', data: dates.map(d => byDate[d].error), backgroundColor: '#d29922' },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, ticks: { color: ct.text2 }, grid: { color: ct.grid, lineWidth: 1, borderDash: [4, 4] } },
          y: { stacked: true, ticks: { color: ct.text2 }, grid: { color: ct.grid } },
        },
        plugins: { legend: { labels: { color: ct.text } } },
      },
    }
  }, [entries, ct])

  return (
    <div className="chart-card">
      <h3>Pass / Fail / Error per Eval Run</h3>
      <div className="chart-container">
        <Bar data={data} options={options} />
      </div>
    </div>
  )
}
