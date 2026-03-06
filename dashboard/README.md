# Lightspeed Evaluation Dashboard

A web-based dashboard for visualizing, comparing, and managing lightspeed-evaluation results. Built with React 19 and Vite, it provides interactive charts, side-by-side evaluation comparison, PDF export, and the ability to run evaluations directly from the browser.

## Quick Start

```bash
# Prerequisites: oc, Node.js 20.19+ or 22.12+, lightspeed-eval Python package installed
make install

# Start development server with default values
make dev

# Start development server with desired values
LS_EVAL_SYSTEM_CFG_PATH=<path_to_system.yaml> LS_EVAL_DATA_PATH=<path_to_eval.yaml> LS_EVAL_REPORTS_PATH=<path_to_reports_dir> LS_EVAL_DASHBOARD_RUN_ENABLED=<true|false> API_KEY="$API_KEY" npx vite
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LS_EVAL_SYSTEM_CFG_PATH` | Yes | Path to `system.yaml` |
| `LS_EVAL_DATA_PATH` | No | Path to `eval.yaml` |
| `LS_EVAL_REPORTS_PATH` | No | Path to reports directory |
| `LS_EVAL_DASHBOARD_RUN_ENABLED` | No | Enable/disable running evaluations from dashboard (default: `true`) |
| `API_KEY` | Yes | API key for OLS service (e.g. `oc whoami -t`) |
| `OPENAI_API_KEY` | For eval | API key for the judge LLM provider |

## Development

### Prerequisites

- **Node.js 20.19+ or 22.12+**
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