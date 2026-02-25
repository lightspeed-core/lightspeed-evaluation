# Lightspeed Evaluation Dashboard

A web-based dashboard for visualizing, comparing, and managing [LightSpeed Evaluation Framework](https://github.com/openshift/lightspeed-service) evaluation results. Built with React 19 and Vite, it provides interactive charts, side-by-side evaluation comparison, PDF export, and the ability to run evaluations directly from the browser.

## Quick Start

```bash
# Prerequisites: Node.js 18+, lightspeed-eval Python package installed

cd eval/web
make install

# Start development server
LS_EVAL_SYSTEM_CFG_PATH=../system.yaml API_KEY=$(oc whoami -t) make dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Features

### Overview Tab
- **Stats Cards** -- total runs, checks, pass/fail/error rates, average score
- **Results Distribution** -- doughnut chart showing PASS / FAIL / ERROR breakdown
- **Metrics Bar Chart** -- pass count per evaluation metric
- **Stacked Bar Chart** -- results grouped by conversation and metric

### Trends Tab
- **Score Trends** -- per-metric scores over time (drag to zoom, click datapoint for details)
- **Average Score** -- rolling average score trend
- **Score Percentiles** -- P10/P25/P50/P75/P90 bands over time
- **Execution Time** -- query execution time patterns

### Results Tab
- Paginated, sortable table of all evaluation results
- Expandable rows with Markdown-rendered reason and response
- Columns: Date, Conversation, Turn, Metric, Score, Result, Query, Reason, Response, Time

### Evaluations Tab
- Browse all evaluation output files with pass/total ratios and per-turn token usage
- **Compare Mode** -- select exactly 2 evaluations to compare side-by-side:
  - Summary pills (improved / regressed / unchanged / total) -- click to filter
  - Delta-colored table with sortable columns and pagination
  - PDF export of comparison results
- **Conversation Config** -- view the YAML configuration used for each evaluation
- **Graphs** -- view generated visualization PNGs (score distribution, status breakdown)
- **PDF Export** -- generate a full report including tables, summary, graphs, and conversation config
- Date range filtering (from / to)

### Conversations Tab
- Browse and edit conversation YAML files
- Run evaluations with a selected conversation
- View git diffs for changed conversations

### Run Tab
- Start new evaluations from the browser
- Real-time terminal output streaming (Server-Sent Events)
- Detects externally-running `lightspeed-eval` processes
- Stop running evaluations, download logs

### System Config
- View and edit `system.yaml` in-browser with syntax highlighting
- Git diff viewer for pending changes

### UI/UX
- **Dark / Light mode** with glassmorphism design
- **Responsive layout** with collapsible chart panels
- **Interactive charts** with zoom, pan, and tooltips
- **Git branch** display in the header

## Project Structure

```
eval/web/
├── src/
│   ├── App.jsx                  # Main app, tab routing, header, modals
│   ├── App.css                  # Layout, theming, glassmorphism styles
│   ├── index.css                # CSS variables (dark/light palettes)
│   ├── main.jsx                 # React entry point
│   ├── hooks/
│   │   ├── useTheme.jsx         # Dark/light toggle, persisted to localStorage
│   │   ├── useEvalData.js       # Fetch & parse evaluation CSV data
│   │   └── useFilters.js        # Filter state (conversation, turn, metric, result, time)
│   └── components/
│       ├── Explorer.jsx         # File browser, compare mode, PDF export
│       ├── RunPage.jsx          # Run evaluations, stream output
│       ├── ResultsTable.jsx     # Paginated sortable results table
│       ├── DetailModal.jsx      # Full evaluation detail on chart click
│       ├── FilterBar.jsx        # Dropdown filters
│       ├── StatsCards.jsx       # KPI stat cards
│       ├── CollapsiblePanel.jsx # Expandable chart container
│       ├── YamlEditor.jsx       # YAML editor with line numbers & diff view
│       ├── ChartSetup.js        # Chart.js plugin registration & defaults
│       ├── ResultsPieChart.jsx  # Doughnut chart
│       ├── MetricBarChart.jsx   # Bar chart by metric
│       ├── StackedBarChart.jsx  # Stacked bar by conversation
│       ├── ScoreTrendChart.jsx  # Score line chart with zoom
│       ├── AvgScoreTrendChart.jsx
│       ├── ExecTimeTrendChart.jsx
│       └── PercentilesChart.jsx
├── vite.config.js               # Vite config + embedded API server
├── package.json
├── Makefile
└── index.html
```

## API Endpoints

All API endpoints are served by the Vite dev server middleware defined in `vite.config.js`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/manifest` | List evaluation CSV files |
| GET | `/api/amended-files` | List amended conversation YAML files |
| GET | `/api/eval-graphs` | Map timestamps to graph PNGs |
| GET | `/api/git-info` | Git branch and repo name |
| GET | `/api/run-config` | System config path, API key status, conversation list |
| GET | `/api/system-config` | Read system.yaml content |
| POST | `/api/system-config` | Save system.yaml content |
| GET | `/api/system-config-diff` | Git diff for system.yaml |
| GET | `/api/conversations` | List conversation YAML files |
| GET | `/api/conversation-content/:path` | Read conversation file |
| POST | `/api/conversation-content/:path` | Save conversation file |
| GET | `/api/conversation-diff/:path` | Git diff for conversation file |
| POST | `/api/run-eval` | Start a new evaluation run |
| GET | `/api/running-evals` | List active evaluations (web + external) |
| POST | `/api/stop-eval/:id` | Stop a running evaluation |
| GET | `/api/eval-stream/:id` | SSE stream of evaluation output |
| GET | `/results/*` | Serve CSV, YAML, and PNG files |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LS_EVAL_SYSTEM_CFG_PATH` | Yes | Path to `system.yaml` (relative to `eval/` or absolute) |
| `API_KEY` | Yes | API key for OLS service (e.g. `oc whoami -t`) |
| `OPENAI_API_KEY` | For eval | API key for the judge LLM provider |

## Development

### Prerequisites

- **Node.js 18+**
- **Python 3.11+** with `lightspeed-eval` installed (for running evaluations)
- **OLS service** running at the endpoint configured in `system.yaml`

### Commands

```bash
make install     # Install npm dependencies
make dev         # Start Vite dev server (port 5173)
npm run build    # Production build to dist/
npm run preview  # Preview production build
npm run lint     # Run ESLint
```

### File Conventions

- **Evaluation CSVs**: `evaluation_YYYYMMDD_HHMMSS_detailed.csv` in `results/`
- **Amended configs**: `*_amended_YYYYMMDD_HHMMSS.yaml` in `results/`
- **Graph PNGs**: `evaluation_YYYYMMDD_HHMMSS_*.png` in `results/graphs/`
- **Conversations**: `.yaml` files in `conversations/`

Files are matched by their embedded timestamp (`YYYYMMDD_HHMMSS`). An evaluation CSV, its amended config, and its graphs all share the same timestamp.

### Adding a New Chart

1. Create a component in `src/components/` (see `ResultsPieChart.jsx` for a minimal example)
2. Use `useChartTheme()` from `useTheme.jsx` for theme-aware colors
3. Import and render in `App.jsx` under the appropriate tab
4. Wrap in `<CollapsiblePanel>` for consistent styling

### Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| React | 19.2 | UI framework |
| Vite | 7.3 | Build tool + dev server |
| Chart.js | 4.5 | Charts and visualizations |
| chartjs-plugin-zoom | 2.2 | Drag-to-zoom on charts |
| PapaParse | 5.5 | CSV parsing |
| jsPDF | 4.1 | PDF generation |
| jspdf-autotable | 5.0 | PDF table formatting |
| react-markdown | 10.1 | Markdown rendering in expandable rows |
| react-syntax-highlighter | 16.1 | YAML syntax highlighting |
| date-fns | 4.1 | Date formatting and adapters |

## Related

- [lightspeed-service](https://github.com/openshift/lightspeed-service) -- the OLS backend
- [lightspeed-evaluation](https://github.com/lightspeed-core/lightspeed-evaluation) -- the evaluation framework
- [eval/README.md](../README.md) -- evaluation setup and dataset documentation
