import { useState, useMemo, useCallback } from 'react'

const ALL = '__all__'

export function useFilters(entries, modelMap) {
  const [model, setModel] = useState(ALL)
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
    const models = [...new Set(afterTurn.map(e => modelMap?.[e.date]).filter(Boolean))].sort()

    const afterModel = model === ALL
      ? afterTurn
      : afterTurn.filter(e => modelMap?.[e.date] === model)
    const metrics = [...new Set(afterModel.map(e => e.metric).filter(Boolean))].sort()

    const afterMetric = metric === ALL
      ? afterModel
      : afterModel.filter(e => e.metric === metric)
    const results = [...new Set(afterMetric.map(e => e.result).filter(Boolean))].sort()

    return { models, groups, turns, metrics, results }
  }, [entries, modelMap, model, group, turn, metric, timeWindow])

  // Auto-reset downstream filters when upstream selection invalidates them
  const wrappedSetGroup = useCallback((v) => {
    setGroup(v)
    setTurn(ALL)
    setModel(ALL)
    setMetric(ALL)
    setResult(ALL)
  }, [])

  const wrappedSetTurn = useCallback((v) => {
    setTurn(v)
    setModel(ALL)
    setMetric(ALL)
    setResult(ALL)
  }, [])

  const wrappedSetModel = useCallback((v) => {
    setModel(v)
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
      (model === ALL || modelMap?.[e.date] === model) &&
      (group === ALL || e.conversationGroupId === group) &&
      (turn === ALL || e.turnId === turn) &&
      (metric === ALL || e.metric === metric) &&
      (result === ALL || e.result === result) &&
      (!cutoff || new Date(e.date) >= cutoff)
    )
  }, [entries, modelMap, model, group, turn, metric, result, timeWindow])

  // Filtered without metric/result — used by AvgScoreTrendChart so metric selection
  // does not affect the average score calculation
  const filteredNoMetric = useMemo(() => {
    let cutoff = null
    if (timeWindow !== 'all') {
      const minutes = parseInt(timeWindow, 10)
      cutoff = new Date(Date.now() - minutes * 60000)
    }
    return entries.filter(e =>
      (model === ALL || modelMap?.[e.date] === model) &&
      (group === ALL || e.conversationGroupId === group) &&
      (turn === ALL || e.turnId === turn) &&
      (!cutoff || new Date(e.date) >= cutoff)
    )
  }, [entries, modelMap, model, group, turn, timeWindow])

  const reset = useCallback(() => {
    setModel(ALL)
    setGroup(ALL)
    setTurn(ALL)
    setMetric(ALL)
    setResult(ALL)
    setTimeWindow('all')
  }, [])

  return {
    filters: { model, group, turn, metric, result, timeWindow },
    setters: {
      setModel: wrappedSetModel,
      setGroup: wrappedSetGroup,
      setTurn: wrappedSetTurn,
      setMetric: wrappedSetMetric,
      setResult,
      setTimeWindow,
    },
    filtered,
    filteredNoMetric,
    availableOptions,
    reset,
    ALL,
  }
}
