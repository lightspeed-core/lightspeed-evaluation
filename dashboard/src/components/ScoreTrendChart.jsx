import { useMemo, useRef, useState, useEffect } from 'react'
import { Line, getElementAtEvent } from 'react-chartjs-2'
import { COLORS, ZOOM_OPTIONS, soloLegendClick, formatDateLabel, createHtmlLegendPlugin } from './ChartSetup'
import { useChartTheme } from '../hooks/useTheme'
import { useMetricMetadata } from '../hooks/useMetricMetadata'

export default function ScoreTrendChart({ entries, onDataClick }) {
  const ct = useChartTheme()
  const chartRef = useRef(null)
  const [zoomed, setZoomed] = useState(false)
  const { metadata: metricMetadata, loading: metadataLoading } = useMetricMetadata()
  const legendId = 'score-trend-legend'
  const metadataRef = useRef(metricMetadata)

  // Keep ref updated with latest metadata
  useEffect(() => {
    metadataRef.current = metricMetadata
  }, [metricMetadata])

  // Force chart update when metadata loads
  useEffect(() => {
    if (!metadataLoading && chartRef.current && Object.keys(metricMetadata).length > 0) {
      chartRef.current.update()
    }
  }, [metricMetadata, metadataLoading])

  const { data, options, plugins, allDates, metricsList } = useMemo(() => {
    const metricsSet = [...new Set(entries.map(e => e.metric))].sort()

    // Build shared date axis across all metrics
    const dateSet = new Set()
    metricsSet.forEach(metric => {
      entries.forEach(e => {
        if (e.metric === metric && e.score !== null) dateSet.add(e.date)
      })
    })
    const sortedDates = [...dateSet].sort()
    const labels = sortedDates.map(formatDateLabel)

    const datasets = metricsSet.map((metric, i) => {
      const byDate = {}
      entries.forEach(e => {
        if (e.metric === metric && e.score !== null) {
          if (!byDate[e.date]) byDate[e.date] = []
          byDate[e.date].push(e.score)
        }
      })
      return {
        label: metric,
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
      metricsList: metricsSet,
      data: { labels, datasets },
      plugins: [createHtmlLegendPlugin(legendId, metadataRef)],
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
            display: false, // Hide default legend, use HTML legend instead
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
                if (label) label += ': '
                if (context.parsed.y !== null) label += context.parsed.y.toFixed(1) + '%'
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
  }, [entries, ct])

  const resetZoom = () => {
    chartRef.current?.resetZoom()
    setZoomed(false)
  }

  const handleClick = (event) => {
    if (!chartRef.current || !onDataClick) return
    const elements = getElementAtEvent(chartRef.current, event)
    if (!elements.length) return
    const { datasetIndex, index } = elements[0]
    const metric = metricsList[datasetIndex]
    onDataClick({ date: allDates[index], metric })
  }

  return (
    <>
      <div id={legendId} style={{ marginBottom: '10px' }}></div>
      <div className="chart-container">
        <Line ref={chartRef} data={data} options={options} plugins={plugins} onClick={handleClick} />
      </div>
      {zoomed && (
        <button className="reset-zoom-btn" onClick={resetZoom}>Reset Zoom</button>
      )}
    </>
  )
}
