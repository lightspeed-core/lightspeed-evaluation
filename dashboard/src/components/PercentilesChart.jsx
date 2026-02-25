import { useMemo, useRef, useState } from 'react'
import { Line } from 'react-chartjs-2'
import { ZOOM_OPTIONS, soloLegendClick } from './ChartSetup'
import { useChartTheme } from '../hooks/useTheme'

function percentile(sorted, p) {
  if (!sorted.length) return 0
  const idx = (p / 100) * (sorted.length - 1)
  const lower = Math.floor(idx)
  const upper = Math.ceil(idx)
  if (lower === upper) return sorted[lower]
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (idx - lower)
}

export default function PercentilesChart({ entries }) {
  const ct = useChartTheme()
  const chartRef = useRef(null)
  const [zoomed, setZoomed] = useState(false)

  const { data, options } = useMemo(() => {
    const scored = entries.filter(e => e.score !== null)
    const byDate = {}
    scored.forEach(e => {
      if (!byDate[e.date]) byDate[e.date] = []
      byDate[e.date].push(e.score)
    })
    const dates = Object.keys(byDate).sort()
    dates.forEach(d => byDate[d].sort((a, b) => a - b))

    const pDefs = [
      { p: 25, label: 'P25', color: '#f85149', dash: [6, 3] },
      { p: 50, label: 'P50 (Median)', color: '#d29922', dash: [] },
      { p: 75, label: 'P75', color: '#3fb950', dash: [6, 3] },
      { p: 90, label: 'P90', color: '#58a6ff', dash: [2, 2] },
    ]

    const datasets = pDefs.map(({ p, label, color, dash }) => ({
      label,
      data: dates.map(d => ({ x: d, y: percentile(byDate[d], p) })),
      borderColor: color,
      backgroundColor: color + '18',
      borderDash: dash,
      fill: false,
      tension: 0.3,
      pointRadius: 5,
      pointHoverRadius: 8,
    }))

    // Add a filled band between P25 and P75
    datasets.push({
      label: 'P25-P75 band',
      data: dates.map(d => ({ x: d, y: percentile(byDate[d], 75) })),
      borderColor: 'transparent',
      backgroundColor: '#3fb95018',
      fill: { target: 0, above: '#3fb95018', below: '#3fb95018' },
      pointRadius: 0,
      pointHoverRadius: 0,
    })

    return {
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'time',
            time: { unit: 'day' },
            ticks: { color: ct.text2 },
            grid: { color: ct.grid, lineWidth: 1, borderDash: [4, 4] },
          },
          y: {
            min: 0, max: 1,
            ticks: { color: ct.text2 },
            grid: { color: ct.grid },
          },
        },
        plugins: {
          legend: {
            labels: {
              color: ct.text,
              filter: (item) => item.text !== 'P25-P75 band',
            },
            onClick: soloLegendClick,
          },
          tooltip: {
            callbacks: {
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
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
