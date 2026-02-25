import { useMemo } from 'react'
import { Bar } from 'react-chartjs-2'
import { COLORS } from './ChartSetup'
import { useChartTheme } from '../hooks/useTheme'

export default function MetricBarChart({ entries }) {
  const ct = useChartTheme()

  const { data, options } = useMemo(() => {
    const byMetric = {}
    entries.forEach(e => {
      if (e.score === null) return
      if (!byMetric[e.metric]) byMetric[e.metric] = []
      byMetric[e.metric].push(e.score)
    })
    const labels = Object.keys(byMetric).sort()
    const avgs = labels.map(m => {
      const vals = byMetric[m]
      return vals.reduce((a, b) => a + b, 0) / vals.length
    })
    return {
      data: {
        labels,
        datasets: [{
          label: 'Avg Score',
          data: avgs,
          backgroundColor: labels.map((_, i) => COLORS[i % COLORS.length]),
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        indexAxis: 'y',
        scales: {
          x: { min: 0, max: 1, ticks: { color: ct.text2 }, grid: { color: ct.grid, lineWidth: 1, borderDash: [4, 4] } },
          y: { ticks: { color: ct.text }, grid: { color: ct.grid } },
        },
        plugins: { legend: { display: false } },
      },
    }
  }, [entries, ct])

  return (
    <div className="chart-card">
      <h3>Average Score by Metric</h3>
      <div className="chart-container">
        <Bar data={data} options={options} />
      </div>
    </div>
  )
}
