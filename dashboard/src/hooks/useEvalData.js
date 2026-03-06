import { useState, useEffect, useMemo, useCallback } from 'react'
import Papa from 'papaparse'

function parseDateFromFilename(filename) {
  const match = filename.match(/evaluation_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})_detailed\.csv/)
  if (!match) return null
  const [, y, mo, d, h, mi, s] = match
  return new Date(`${y}-${mo}-${d}T${h}:${mi}:${s}`)
}

async function fetchManifest() {
  const res = await fetch('/api/manifest')
  if (!res.ok) throw new Error(`Failed to fetch manifest (${res.status} ${res.statusText})`)
  return res.json()
}

async function fetchAndParseCsv(filename) {
  const res = await fetch(`/results/${filename}`)
  if (!res.ok) throw new Error(`Failed to fetch ${filename} (${res.status} ${res.statusText})`)
  const text = await res.text()
  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: (results) => resolve(results.data),
      error: (err) => reject(err),
    })
  })
}

export function useEvalData() {
  const [entries, setEntries] = useState([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)
  const [reportsDir, setReportsDir] = useState({ path: '', exists: true })
  const [modelMap, setModelMap] = useState({})
  const [reloadKey, setReloadKey] = useState(0)

  const refresh = useCallback(() => setReloadKey(k => k + 1), [])

  useEffect(() => {
    let cancelled = false
    const isRefresh = reloadKey > 0
    if (isRefresh) setRefreshing(true)
    async function load() {
      if (!cancelled) setError(null)
      try {
        const [manifest, summaries] = await Promise.all([
          fetchManifest(),
          fetch('/api/eval-summaries').then(r => r.json()).catch(() => ({})),
        ])
        const files = manifest.files || manifest
        if (!cancelled) {
          setReportsDir({
            path: manifest.reportsDir || '',
            exists: manifest.reportsDirExists !== false,
          })
        }
        const allEntries = []
        const newModelMap = {}

        for (const file of files) {
          const date = parseDateFromFilename(file)
          if (!date) continue
          const tsMatch = file.match(/(\d{8}_\d{6})/)
          if (tsMatch && summaries[tsMatch[1]]?.model) {
            newModelMap[date.toISOString()] = summaries[tsMatch[1]].model
          }
          const rows = await fetchAndParseCsv(file)
          for (const row of rows) {
            const scoreRaw = (row.score || '').trim()
            allEntries.push({
              date: date.toISOString(),
              file,
              conversationGroupId: row.conversation_group_id || '',
              turnId: row.turn_id || '',
              metric: row.metric_identifier || '',
              score: scoreRaw ? parseFloat(scoreRaw) : null,
              result: row.result || '',
              reason: row.reason || '',
              query: row.query || '',
              response: row.response || '',
              executionTime: row.execution_time?.trim()
                ? parseFloat(row.execution_time)
                : null,
            })
          }
        }

        if (!cancelled) {
          setEntries(allEntries)
          setModelMap(newModelMap)
          setLoading(false)
          setRefreshing(false)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message)
          setLoading(false)
          setRefreshing(false)
        }
      }
    }
    load()
    return () => { cancelled = true }
  }, [reloadKey])

  const metadata = useMemo(() => {
    const dates = [...new Set(entries.map(e => e.date))].sort()
    const metrics = [...new Set(entries.map(e => e.metric))].sort()
    const groups = [...new Set(entries.map(e => e.conversationGroupId))].sort()
    const turns = [...new Set(entries.map(e => e.turnId))].sort()
    return { dates, metrics, groups, turns }
  }, [entries])

  return { entries, loading, refreshing, error, metadata, reportsDir, refresh, modelMap }
}
