import { useState, useMemo, useCallback } from 'react'

const ALL = '__all__'

export function useFilters(entries) {
  const [group, setGroup] = useState(ALL)
  const [turn, setTurn] = useState(ALL)
  const [metric, setMetric] = useState(ALL)
  const [result, setResult] = useState(ALL)
  const [timeWindow, setTimeWindow] = useState('all')

  // Cascading available options: each dropdown is filtered by upstream selections
  const availableOptions = useMemo(() => {
    let cutoff = null
    if (timeWindow !== 'all') {
      const minutes = parseInt(timeWindow, 10)
      cutoff = new Date(Date.now() - minutes * 60000)
    }
    const timeFiltered = cutoff
      ? entries.filter(e => new Date(e.date) >= cutoff)
      : entries

    const groups = [...new Set(timeFiltered.map(e => e.conversationGroupId).filter(Boolean))].sort()

    const afterGroup = group === ALL
      ? timeFiltered
      : timeFiltered.filter(e => e.conversationGroupId === group)
    const turns = [...new Set(afterGroup.map(e => e.turnId).filter(Boolean))].sort()

    const afterTurn = turn === ALL
      ? afterGroup
      : afterGroup.filter(e => e.turnId === turn)
    const metrics = [...new Set(afterTurn.map(e => e.metric).filter(Boolean))].sort()

    const afterMetric = metric === ALL
      ? afterTurn
      : afterTurn.filter(e => e.metric === metric)
    const results = [...new Set(afterMetric.map(e => e.result).filter(Boolean))].sort()

    return { groups, turns, metrics, results }
  }, [entries, group, turn, metric, timeWindow])

  // Auto-reset downstream filters when upstream selection invalidates them
  const wrappedSetGroup = useCallback((v) => {
    setGroup(v)
    setTurn(ALL)
    setMetric(ALL)
    setResult(ALL)
  }, [])

  const wrappedSetTurn = useCallback((v) => {
    setTurn(v)
    setMetric(ALL)
    setResult(ALL)
  }, [])

  const wrappedSetMetric = useCallback((v) => {
    setMetric(v)
    setResult(ALL)
  }, [])

  const filtered = useMemo(() => {
    let cutoff = null
    if (timeWindow !== 'all') {
      const minutes = parseInt(timeWindow, 10)
      cutoff = new Date(Date.now() - minutes * 60000)
    }
    return entries.filter(e =>
      (group === ALL || e.conversationGroupId === group) &&
      (turn === ALL || e.turnId === turn) &&
      (metric === ALL || e.metric === metric) &&
      (result === ALL || e.result === result) &&
      (!cutoff || new Date(e.date) >= cutoff)
    )
  }, [entries, group, turn, metric, result, timeWindow])

  const reset = useCallback(() => {
    setGroup(ALL)
    setTurn(ALL)
    setMetric(ALL)
    setResult(ALL)
    setTimeWindow('all')
  }, [])

  return {
    filters: { group, turn, metric, result, timeWindow },
    setters: {
      setGroup: wrappedSetGroup,
      setTurn: wrappedSetTurn,
      setMetric: wrappedSetMetric,
      setResult,
      setTimeWindow,
    },
    filtered,
    availableOptions,
    reset,
    ALL,
  }
}
