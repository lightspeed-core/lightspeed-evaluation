import { useMemo } from 'react'
import { Doughnut } from 'react-chartjs-2'
import { useChartTheme } from '../hooks/useTheme'

export default function ResultsPieChart({ entries }) {
  const ct = useChartTheme()

  const data = useMemo(() => {
    const pass = entries.filter(e => e.result === 'PASS').length
    const fail = entries.filter(e => e.result === 'FAIL').length
    const err = entries.filter(e => e.result === 'ERROR').length
    return {
      labels: ['PASS', 'FAIL', 'ERROR'],
      datasets: [{
        data: [pass, fail, err],
        backgroundColor: ['#3fb950', '#f85149', '#d29922'],
      }],
    }
  }, [entries])

  return (
    <div className="chart-card">
      <h3>Results Distribution</h3>
      <div className="chart-container">
        <Doughnut
          data={data}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: ct.text } } },
          }}
        />
      </div>
    </div>
  )
}
