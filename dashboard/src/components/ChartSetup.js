import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js'
import zoomPlugin from 'chartjs-plugin-zoom'
import 'chartjs-adapter-date-fns'

ChartJS.register(
  CategoryScale,
  LinearScale,
  TimeScale,
  PointElement,
  LineElement,
  BarElement,
  ArcElement,
  Filler,
  Tooltip,
  Legend,
  zoomPlugin
)

export const ZOOM_OPTIONS = {
  zoom: {
    drag: {
      enabled: true,
      backgroundColor: 'rgba(88,166,255,0.1)',
      borderColor: 'rgba(88,166,255,0.4)',
      borderWidth: 1,
    },
    mode: 'x',
  },
}

export const COLORS = [
  '#58a6ff', '#3fb950', '#f85149', '#d29922', '#bc8cff',
  '#f778ba', '#79c0ff', '#7ee787', '#ffa657', '#ff7b72',
]

export function soloLegendClick(event, legendItem, legend) {
  const chart = legend.chart
  const clickedIndex = legendItem.datasetIndex
  const legendIndices = legend.legendItems.map(item => item.datasetIndex)
  const visibleIndices = legendIndices.filter(i => chart.isDatasetVisible(i))
  const allVisible = visibleIndices.length === legendIndices.length
  const clickedIsVisible = chart.isDatasetVisible(clickedIndex)

  if (allVisible) {
    for (const i of legendIndices) {
      chart.setDatasetVisibility(i, i === clickedIndex)
    }
  } else if (clickedIsVisible && visibleIndices.length === 1) {
    for (const i of legendIndices) {
      chart.setDatasetVisibility(i, true)
    }
  } else {
    chart.setDatasetVisibility(clickedIndex, !clickedIsVisible)
  }

  chart.update()
}

export function formatDateLabel(isoString) {
  const d = new Date(isoString)
  const month = d.toLocaleString('en-US', { month: 'short' })
  const day = String(d.getDate()).padStart(2, '0')
  const hour = String(d.getHours()).padStart(2, '0')
  const min = String(d.getMinutes()).padStart(2, '0')
  return `${month} ${day} ${hour}:${min}`
}

/**
 * Creates a custom HTML legend plugin with tooltips for metric descriptions
 * @param {string} containerId - ID of the container element for the legend
 * @param {Object|Ref} metricMetadataOrRef - Map of metric names to descriptions, or a ref containing it
 * @returns {Object} Chart.js plugin object
 */
export function createHtmlLegendPlugin(containerId, metricMetadataOrRef = {}) {
  return {
    id: `htmlLegend-${containerId}`,
    afterUpdate(chart) {
      const ul = getOrCreateLegendList(chart, containerId)

      // Remove old legend items
      while (ul.firstChild) {
        ul.firstChild.remove()
      }

      // Reuse the built-in legendItems generator
      const items = chart.options.plugins.legend.labels.generateLabels(chart)

      // Get metadata from ref or direct object
      const metricMetadata = metricMetadataOrRef.current || metricMetadataOrRef

      items.forEach((item) => {
        const li = document.createElement('li')
        li.style.alignItems = 'center'
        li.style.cursor = 'pointer'
        li.style.display = 'flex'
        li.style.flexDirection = 'row'
        li.style.marginLeft = '10px'
        li.style.position = 'relative'

        li.onclick = () => {
          const { type } = chart.config
          if (type === 'pie' || type === 'doughnut') {
            chart.toggleDataVisibility(item.index)
          } else {
            // Solo legend click behavior
            const clickedIndex = item.datasetIndex
            const allIndices = items.map(i => i.datasetIndex)
            const visibleIndices = allIndices.filter(i => chart.isDatasetVisible(i))
            const allVisible = visibleIndices.length === allIndices.length
            const clickedIsVisible = chart.isDatasetVisible(clickedIndex)

            if (allVisible) {
              // All visible: hide all except clicked
              for (const i of allIndices) {
                chart.setDatasetVisibility(i, i === clickedIndex)
              }
            } else if (clickedIsVisible && visibleIndices.length === 1) {
              // Only clicked is visible: show all
              for (const i of allIndices) {
                chart.setDatasetVisibility(i, true)
              }
            } else {
              // Toggle clicked item
              chart.setDatasetVisibility(clickedIndex, !clickedIsVisible)
            }
          }
          chart.update()
        }

        // Color box
        const boxSpan = document.createElement('span')
        boxSpan.style.background = item.fillStyle
        boxSpan.style.borderColor = item.strokeStyle
        boxSpan.style.borderWidth = item.lineWidth + 'px'
        boxSpan.style.display = 'inline-block'
        boxSpan.style.flexShrink = '0'
        boxSpan.style.height = '20px'
        boxSpan.style.marginRight = '10px'
        boxSpan.style.width = '20px'

        // Text
        const textContainer = document.createElement('p')
        textContainer.style.color = item.fontColor
        textContainer.style.margin = '0'
        textContainer.style.padding = '0'
        textContainer.style.textDecoration = item.hidden ? 'line-through' : ''

        const text = document.createTextNode(item.text)
        textContainer.appendChild(text)

        // Add tooltip if description exists
        const description = metricMetadata[item.text]
        if (description) {
          li.title = description
          li.style.cursor = 'help'
        } else {
          // Fallback: show metric name as tooltip if no description
          li.title = `Metric: ${item.text} (no description available)`
        }

        li.appendChild(boxSpan)
        li.appendChild(textContainer)
        ul.appendChild(li)
      })
    },
  }
}

function getOrCreateLegendList(chart, id) {
  const legendContainer = document.getElementById(id)
  let listContainer = legendContainer.querySelector('ul')

  if (!listContainer) {
    listContainer = document.createElement('ul')
    listContainer.style.display = 'flex'
    listContainer.style.flexDirection = 'row'
    listContainer.style.flexWrap = 'wrap'
    listContainer.style.margin = '0'
    listContainer.style.padding = '0'
    listContainer.style.listStyle = 'none'
    legendContainer.appendChild(listContainer)
  }

  return listContainer
}

export const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: { labels: { color: '#e6edf3' } },
  },
  scales: {
    x: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
  },
}
