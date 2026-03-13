import { useMemo, useRef, useState } from 'react'
import { Line } from 'react-chartjs-2'
import { ZOOM_OPTIONS, formatDateLabel, createHtmlLegendPlugin } from './ChartSetup'
import { useChartTheme } from '../hooks/useTheme'

export default function ExecTimeTrendChart({ entries }) {
  const ct = useChartTheme()
  const chartRef = useRef(null)
  const [zoomed, setZoomed] = useState(false)
  const legendId = 'exec-time-legend'

  const { data, options, plugins } = useMemo(() => {
    const withTime = entries.filter(e => e.executionTime !== null)
    const byDate = {}
    withTime.forEach(e => {
      if (!byDate[e.date]) byDate[e.date] = []
      byDate[e.date].push(e.executionTime)
    })
    const sortedDates = Object.keys(byDate).sort()
    const labels = sortedDates.map(formatDateLabel)
    return {
      data: {
        labels,
        datasets: [{
          label: 'Avg Execution Time (s)',
          data: sortedDates.map(d => byDate[d].reduce((a, b) => a + b, 0) / byDate[d].length),
          borderColor: '#bc8cff',
          backgroundColor: '#bc8cff33',
          fill: true,
          tension: 0.3,
          pointRadius: 5,
          pointHoverRadius: 8,
        }],
      },
      plugins: [createHtmlLegendPlugin(legendId, {})],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: { color: ct.text2, maxRotation: 45 },
            grid: { color: ct.grid, lineWidth: 1, borderDash: [4, 4] },
          },
          y: {
            min: 0,
            ticks: { color: ct.text2 },
            grid: { color: ct.grid },
          },
        },
        plugins: {
          legend: {
            display: false,
            labels: {
              color: ct.text,
              generateLabels: (chart) => {
                const datasets = chart.data.datasets
                return datasets.map((dataset, i) => ({
                  text: dataset.label,
                  fillStyle: dataset.borderColor,
                  strokeStyle: dataset.borderColor,
                  lineWidth: 2,
                  hidden: !chart.isDatasetVisible(i),
                  index: i,
                  datasetIndex: i,
                  fontColor: ct.text,
                }))
              }
            }
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
      <div id={legendId} style={{ marginBottom: '10px', minHeight: '30px' }}></div>
      <div className="chart-container">
        <Line ref={chartRef} data={data} options={options} plugins={plugins} />
      </div>
      {zoomed && (
        <button className="reset-zoom-btn" onClick={resetZoom}>Reset Zoom</button>
      )}
    </>
  )
}
