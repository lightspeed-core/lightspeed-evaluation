import { useState, useEffect } from 'react'
import yaml from 'js-yaml'

/**
 * Fetches and parses metric metadata from system.yaml
 * Returns a map of metric identifiers to their descriptions
 */
export function useMetricMetadata() {
  const [metadata, setMetadata] = useState({})
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/api/system-config')
      .then(r => r.json())
      .then(data => {
        if (!data.content) {
          setLoading(false)
          return
        }
        try {
          const config = yaml.load(data.content)
          const metricMap = {}

          // Parse turn-level metrics
          if (config.metrics_metadata?.turn_level) {
            Object.entries(config.metrics_metadata.turn_level).forEach(([key, value]) => {
              if (value.description) {
                metricMap[key] = value.description
              }
            })
          }

          // Parse conversation-level metrics
          if (config.metrics_metadata?.conversation_level) {
            Object.entries(config.metrics_metadata.conversation_level).forEach(([key, value]) => {
              if (value.description) {
                metricMap[key] = value.description
              }
            })
          }

          setMetadata(metricMap)
        } catch (err) {
          console.error('Failed to parse system config:', err)
        }
        setLoading(false)
      })
      .catch((err) => {
        console.error('Failed to fetch system config:', err)
        setLoading(false)
      })
  }, [])

  return { metadata, loading }
}
