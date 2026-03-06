import { useState, useEffect } from 'react'
import { Light as SyntaxHighlighter } from 'react-syntax-highlighter'
import yamlLang from 'react-syntax-highlighter/dist/esm/languages/hljs/yaml'
import { atomOneDark, atomOneLight } from 'react-syntax-highlighter/dist/esm/styles/hljs'
import './components/ChartSetup'
import { useEvalData } from './hooks/useEvalData'
import { useFilters } from './hooks/useFilters'
import { useTheme } from './hooks/useTheme'
import FilterBar from './components/FilterBar'
import StatsCards from './components/StatsCards'
import ResultsPieChart from './components/ResultsPieChart'
import MetricBarChart from './components/MetricBarChart'
import StackedBarChart from './components/StackedBarChart'
import ScoreTrendChart from './components/ScoreTrendChart'
import OverallAvgChart from './components/OverallAvgChart'
import AvgScoreTrendChart from './components/AvgScoreTrendChart'
import ExecTimeTrendChart from './components/ExecTimeTrendChart'
import CollapsiblePanel from './components/CollapsiblePanel'
import DetailModal from './components/DetailModal'
import ResultsTable from './components/ResultsTable'
import Evaluations from './components/Evaluations'
import RunPage from './components/RunPage'
import './App.css'

SyntaxHighlighter.registerLanguage('yaml', yamlLang)

const TAB_ICONS = {
  overview: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></svg>,
  evaluations: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /></svg>,
  conversations: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" /><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" /></svg>,
  trends: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12" /></svg>,
  details: <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 11 12 14 22 4" /><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" /></svg>,
}

const TAB_GROUPS = [
  [
    { id: 'trends', label: 'Overview' },
    { id: 'overview', label: 'Insights' },
  ],
  [
    { id: 'evaluations', label: 'Evaluations' },
    { id: 'conversations', label: 'Conversations' },
    { id: 'details', label: 'Details' },
  ],
]

function SystemConfigModal({ config, onClose }) {
  const { theme } = useTheme()

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content amended-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>System Configuration</h2>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body">
          <SyntaxHighlighter
            language="yaml"
            style={theme === 'dark' ? atomOneDark : atomOneLight}
            customStyle={{
              margin: 0,
              borderRadius: '6px',
              fontSize: '13px',
              lineHeight: '1.6',
            }}
            showLineNumbers
          >
            {config.content}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  )
}

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  return (
    <button
      className="theme-toggle"
      onClick={toggle}
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
      aria-label="Toggle theme"
    >
      {theme === 'dark' ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="5"/>
          <line x1="12" y1="1" x2="12" y2="3"/>
          <line x1="12" y1="21" x2="12" y2="23"/>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
          <line x1="1" y1="12" x2="3" y2="12"/>
          <line x1="21" y1="12" x2="23" y2="12"/>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
        </svg>
      )}
    </button>
  )
}

function EvalDataModal({ config, onClose }) {
  const { theme } = useTheme()

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content amended-modal" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h2>Evaluation Data</h2>
          </div>
          <button className="modal-close" onClick={onClose} aria-label="Close">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="modal-body">
          <SyntaxHighlighter
            language="yaml"
            style={theme === 'dark' ? atomOneDark : atomOneLight}
            customStyle={{
              margin: 0,
              borderRadius: '6px',
              fontSize: '13px',
              lineHeight: '1.6',
            }}
            showLineNumbers
          >
            {config.content}
          </SyntaxHighlighter>
        </div>
      </div>
    </div>
  )
}

export default function App() {
  const { entries, loading, refreshing, error, reportsDir, refresh, modelMap } = useEvalData()
  const { filters, setters, filtered, filteredNoMetric, availableOptions, reset, ALL } = useFilters(entries, modelMap)
  const [activeTab, setActiveTab] = useState('trends')
  const [refreshKey, setRefreshKey] = useState(0)
  const [detailView, setDetailView] = useState(null)
  const [systemConfig, setSystemConfig] = useState({ set: false, path: '', content: '' })
  const [showSystemConfig, setShowSystemConfig] = useState(false)
  const [evalDataConfig, setEvalDataConfig] = useState(null)
  const [evalDataLoading, setEvalDataLoading] = useState(false)

  useEffect(() => {
    fetch('/api/system-config').then(r => r.json())
      .then(data => setSystemConfig(data))
      .catch(() => {})
  }, [])

  const openEvalData = async () => {
    setEvalDataLoading(true)
    try {
      const res = await fetch('/api/eval-data-content')
      if (!res.ok) {
        console.error('Failed to load eval data:', res.status)
        setEvalDataConfig(null)
        return
      }
      const data = await res.json()
      setEvalDataConfig({ path: data.path, content: data.content })
    } catch {
      setEvalDataConfig(null)
    } finally {
      setEvalDataLoading(false)
    }
  }



  if (loading) {
    return (
      <div className="loading">
        <div className="spinner" />
        <p>Loading evaluation data...</p>
      </div>
    )
  }

  if (error) {
    return <div className="error-msg">Failed to load data: {error}</div>
  }

  return (
    <div className="app-wrapper">
      <header>
        <h1 onClick={() => setActiveTab('trends')} style={{ cursor: 'pointer' }}>
          Lightspeed Evaluation Dashboard
        </h1>
        <span className="subtitle" />
        {systemConfig.runEnabled && (
          <button
            className={`system-config-btn${systemConfig.set ? '' : ' disabled'}`}
            onClick={() => systemConfig.set && setShowSystemConfig(true)}
            disabled={!systemConfig.set}
            title={systemConfig.set ? systemConfig.path : 'LS_EVAL_SYSTEM_CFG_PATH not set'}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            {systemConfig.set ? 'System Config' : 'No System Config'}
          </button>
        )}
        {systemConfig.runEnabled && (
          <button
            className="system-config-btn"
            onClick={openEvalData}
            title="View eval.yaml"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
            </svg>
            Eval Data
          </button>
        )}
        <ThemeToggle />
      </header>

      <div className="container">
        <div className="tabs" role="tablist">
          {TAB_GROUPS.map((group, gi) => (
            <div key={gi} className="tab-group">
              {gi > 0 && <div className="tab-divider" />}
              {group.map(t => (
                <button
                  key={t.id}
                  type="button"
                  role="tab"
                  aria-selected={activeTab === t.id}
                  tabIndex={activeTab === t.id ? 0 : -1}
                  className={`tab ${activeTab === t.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(t.id)}
                  onKeyDown={(e) => {
                    const allTabs = TAB_GROUPS.flat()
                    const idx = allTabs.findIndex(tab => tab.id === t.id)
                    let nextIdx = -1
                    if (e.key === 'ArrowRight') nextIdx = (idx + 1) % allTabs.length
                    else if (e.key === 'ArrowLeft') nextIdx = (idx - 1 + allTabs.length) % allTabs.length
                    else if (e.key === 'Home') nextIdx = 0
                    else if (e.key === 'End') nextIdx = allTabs.length - 1
                    if (nextIdx >= 0) {
                      e.preventDefault()
                      setActiveTab(allTabs[nextIdx].id)
                      e.currentTarget.parentElement.closest('[role="tablist"]')
                        ?.querySelectorAll('[role="tab"]')[nextIdx]?.focus()
                    }
                  }}
                >
                  {TAB_ICONS[t.id]}{t.label}
                </button>
              ))}
            </div>
          ))}
          <div className="tab-spacer" />
          <button
            className={`refresh-btn${refreshing ? ' spinning' : ''}`}
            onClick={() => { refresh(); setRefreshKey(k => k + 1) }}
            disabled={refreshing}
            title="Refresh data"
            aria-label="Refresh data"
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="23 4 23 10 17 10" />
              <polyline points="1 20 1 14 7 14" />
              <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
            </svg>
          </button>
          {systemConfig.runEnabled && (
            <button
              className={`run-btn${activeTab === 'run' ? ' active' : ''}`}
              onClick={() => setActiveTab('run')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              Run
            </button>
          )}
        </div>

        <div className="tab-content">
          {!['evaluations', 'conversations', 'run'].includes(activeTab) && (
            <FilterBar
              availableOptions={availableOptions}
              filters={filters}
              setters={setters}
              reset={reset}
              ALL={ALL}
            />
          )}

          {activeTab === 'overview' && (
            <>
              <StatsCards entries={filtered} />
              <div className="charts two-col">
                <ResultsPieChart entries={filtered} />
                <MetricBarChart entries={filtered} />
              </div>
              <div className="charts">
                <StackedBarChart entries={filtered} />
              </div>
            </>
          )}

          {activeTab === 'evaluations' && (
            <Evaluations reportsDir={reportsDir} view="evaluations" refreshKey={refreshKey} />
          )}

          {activeTab === 'conversations' && (
            <Evaluations view="conversations" refreshKey={refreshKey} />
          )}

          {activeTab === 'trends' && (
            <div className="charts">
              <CollapsiblePanel title="Overall Average Score">
                <OverallAvgChart entries={entries} />
              </CollapsiblePanel>
              <CollapsiblePanel title="Average Score Over Time" tooltip="Click on a datapoint to view full evaluation details">
                <AvgScoreTrendChart entries={filteredNoMetric} modelMap={modelMap} onDataClick={setDetailView} />
              </CollapsiblePanel>
              <CollapsiblePanel title="Score Trends Over Time (by Metric)" tooltip="Click on a datapoint to view full evaluation details">
                <ScoreTrendChart entries={filtered} onDataClick={setDetailView} />
              </CollapsiblePanel>
              <CollapsiblePanel title="Execution Time Trends">
                <ExecTimeTrendChart entries={filtered} />
              </CollapsiblePanel>
            </div>
          )}

          {activeTab === 'details' && (
            <ResultsTable entries={filtered} />
          )}

          {activeTab === 'run' && systemConfig.runEnabled && (
            <RunPage />
          )}
        </div>
      </div>

      {detailView && (
        <DetailModal
          date={detailView.date}
          metric={detailView.metric}
          entries={filtered}
          modelMap={modelMap}
          onClose={() => setDetailView(null)}
        />
      )}

      {showSystemConfig && systemConfig.set && (
        <SystemConfigModal
          config={systemConfig}
          onClose={() => setShowSystemConfig(false)}
        />
      )}

      {evalDataConfig && (
        <EvalDataModal
          config={evalDataConfig}
          onClose={() => setEvalDataConfig(null)}
        />
      )}

      {evalDataLoading && (
        <div className="modal-overlay">
          <div className="explorer-loading">
            <div className="spinner" />
            <p>Loading eval data...</p>
          </div>
        </div>
      )}

    </div>
  )
}
