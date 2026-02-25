export default function FilterBar({ availableOptions, filters, setters, reset, ALL }) {
  return (
    <div className="filters">
      <div className="filter-group">
        <label>Conversation</label>
        <select value={filters.group} onChange={e => setters.setGroup(e.target.value)}>
          <option value={ALL}>All Conversations</option>
          {availableOptions.groups.map(g => <option key={g} value={g}>{g}</option>)}
        </select>
      </div>
      <div className="filter-group">
        <label>Turn</label>
        <select value={filters.turn} onChange={e => setters.setTurn(e.target.value)}>
          <option value={ALL}>All Turns</option>
          {availableOptions.turns.map(t => <option key={t} value={t}>{t}</option>)}
        </select>
      </div>
      <div className="filter-group">
        <label>Metric</label>
        <select value={filters.metric} onChange={e => setters.setMetric(e.target.value)}>
          <option value={ALL}>All Metrics</option>
          {availableOptions.metrics.map(m => <option key={m} value={m}>{m}</option>)}
        </select>
      </div>
      <div className="filter-group">
        <label>Result</label>
        <select value={filters.result} onChange={e => setters.setResult(e.target.value)}>
          <option value={ALL}>All</option>
          {availableOptions.results.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>
      <div className="filter-group">
        <label>Time Window</label>
        <select value={filters.timeWindow} onChange={e => setters.setTimeWindow(e.target.value)}>
          <option value="all">All Time</option>
          <option value="5">Last 5 Min</option>
          <option value="30">Last 30 Min</option>
          <option value="60">Last 1 Hour</option>
          <option value="360">Last 6 Hours</option>
          <option value="1440">Last 1 Day</option>
          <option value="10080">Last 7 Days</option>
          <option value="20160">Last 14 Days</option>
          <option value="43200">Last 30 Days</option>
          <option value="129600">Last 90 Days</option>
        </select>
      </div>
      <div className="filter-group">
        <label>&nbsp;</label>
        <button onClick={reset}>Reset</button>
      </div>
    </div>
  )
}
