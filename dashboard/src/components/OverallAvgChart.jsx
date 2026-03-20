import { useMemo, useRef, useState } from 'react'
import { Line } from 'react-chartjs-2'
import { ZOOM_OPTIONS, formatDateLabel } from './ChartSetup'
import { useChartTheme } from '../hooks/useTheme'

export default function OverallAvgChart({ entries }) {
  const ct = useChartTheme()
  const chartRef = useRef(null)
  const [zoomed, setZoomed] = useState(false)

  const { data, options } = useMemo(() => {
    // Group all scores by date
    const byDate = {}
    entries.forEach(e => {
      if (e.score === null) return
      if (!byDate[e.date]) byDate[e.date] = []
      byDate[e.date].push(e.score)
    })

    const sortedDates = Object.keys(byDate).sort()
    const labels = sortedDates.map(formatDateLabel)
    const avgData = sortedDates.map(d => {
      const scores = byDate[d]
      return (scores.reduce((a, b) => a + b, 0) / scores.length) * 100
    })

    return {
      data: {
        labels,
        datasets: [{
          label: 'Overall Average',
          data: avgData,
          borderColor: '#58a6ff',
          backgroundColor: 'rgba(88, 166, 255, 0.1)',
          fill: true,
          tension: 0.3,
          pointRadius: 5,
          pointHoverRadius: 8,
          spanGaps: true,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: { color: ct.text2, maxRotation: 45 },
            grid: { color: ct.grid, lineWidth: 1, borderDash: [4, 4] },
          },
          y: {
            min: 0, max: 100,
            ticks: {
              color: ct.text2,
              callback: function(value) {
                return value + '%'
              },
            },
            grid: { color: ct.grid },
          },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || ''
                if (label) label += ': '
                if (context.parsed.y !== null) label += context.parsed.y.toFixed(1) + '%'
                return label
              },
            },
          },
          zoom: {
            ...ZOOM_OPTIONS,
            zoom: {
              ...ZOOM_OPTIONS.zoom,
              onZoomComplete: () => setZoomed(true),
            },
          },
        },
      },
    }
  }, [entries, ct])

  const resetZoom = () => {
    chartRef.current?.resetZoom()
    setZoomed(false)
  }

  return (
    <>
      <div className="chart-container">
        <Line ref={chartRef} data={data} options={options} />
      </div>
      {zoomed && (
        <button className="reset-zoom-btn" onClick={resetZoom}>Reset Zoom</button>
      )}
    </>
  )
}
