import { useMemo, useRef, useState } from 'react'
import { Line, getElementAtEvent } from 'react-chartjs-2'
import { COLORS, ZOOM_OPTIONS, soloLegendClick, formatDateLabel, createHtmlLegendPlugin } from './ChartSetup'
import { useChartTheme } from '../hooks/useTheme'

export default function AvgScoreTrendChart({ entries, modelMap, onDataClick }) {
  const ct = useChartTheme()
  const chartRef = useRef(null)
  const [zoomed, setZoomed] = useState(false)
  const legendId = 'avg-score-legend'

  const { data, options, plugins, allDates, conversationList } = useMemo(() => {
    // Build series keys combining conversation + model
    const seriesSet = new Set()
    entries.forEach(e => {
      if (!e.conversationGroupId || e.score === null) return
      const model = modelMap?.[e.date]
      const key = model ? `${e.conversationGroupId} (${model})` : e.conversationGroupId
      seriesSet.add(key)
    })
    const seriesKeys = [...seriesSet].sort()

    // Build shared date axis
    const dateSet = new Set()
    entries.forEach(e => {
      if (e.score !== null) dateSet.add(e.date)
    })
    const sortedDates = [...dateSet].sort()
    const labels = sortedDates.map(formatDateLabel)

    const datasets = seriesKeys.map((seriesKey, i) => {
      const byDate = {}
      entries.forEach(e => {
        if (e.score === null || !e.conversationGroupId) return
        const model = modelMap?.[e.date]
        const key = model ? `${e.conversationGroupId} (${model})` : e.conversationGroupId
        if (key === seriesKey) {
          if (!byDate[e.date]) byDate[e.date] = []
          byDate[e.date].push(e.score)
        }
      })
      return {
        label: seriesKey,
        data: sortedDates.map(d =>
          byDate[d] ? (byDate[d].reduce((a, b) => a + b, 0) / byDate[d].length) * 100 : null
        ),
        borderColor: COLORS[i % COLORS.length],
        backgroundColor: COLORS[i % COLORS.length] + '33',
        fill: false,
        tension: 0.3,
        pointRadius: 5,
        pointHoverRadius: 8,
        spanGaps: true,
      }
    })

    return {
      allDates: sortedDates,
      conversationList: seriesKeys,
      data: { labels, datasets },
      plugins: [createHtmlLegendPlugin(legendId, {})],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        onHover: (event, elements) => {
          event.native.target.style.cursor = elements.length ? 'pointer' : 'default'
        },
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
              }
            },
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
            },
            onClick: soloLegendClick
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                let label = context.dataset.label || ''
                if (label) {
                  label += ': '
                }
                if (context.parsed.y !== null) {
                  label += context.parsed.y.toFixed(1) + '%'
                }
                return label
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
  }, [entries, modelMap, ct])

  const resetZoom = () => {
    chartRef.current?.resetZoom()
    setZoomed(false)
  }

  const handleClick = (event) => {
    if (!chartRef.current || !onDataClick) return
    const elements = getElementAtEvent(chartRef.current, event)
    if (!elements.length) return
    const { index } = elements[0]
    onDataClick({ date: allDates[index], metric: null })
  }

  return (
    <>
      <div id={legendId} style={{ marginBottom: '10px', minHeight: '30px' }}></div>
      <div className="chart-container">
        <Line ref={chartRef} data={data} options={options} plugins={plugins} onClick={handleClick} />
      </div>
      {zoomed && (
        <button className="reset-zoom-btn" onClick={resetZoom}>Reset Zoom</button>
      )}
    </>
  )
}
