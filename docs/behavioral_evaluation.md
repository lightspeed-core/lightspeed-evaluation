# Behavioral Evaluation (MxN)

MxN behavioral evaluation compares **M live agent configurations** across **N repetitions** of the same evaluation cases. It runs the M×N matrix, aggregates results, and writes per-agent and cross-agent reports.

This is a **live evaluation**, not an offline workflow. Configure and start each HTTP API before evaluation. For `openshift_agentic_run`, configure cluster access and ensure Kubernetes API is reachable.

## Configure agents

Set `agents.enabled: true`, list agent names in `agents.default.agent`, and set repetition and concurrency options. Configure each `api_base` for a running API instance.

```yaml
agents:
  enabled: true
  default:
    agent: [baseline, candidate]
    repeat: 3
    parallel: false

  baseline:
    type: http_api
    api_base: http://localhost:8080
    endpoint_type: streaming

  candidate:
    type: http_api
    api_base: http://localhost:8081
    endpoint_type: streaming
```

`repeat` must be positive. `parallel: true` runs separate processes concurrently; ensure both APIs and host have enough capacity. `parallel: false` runs sequentially.

The default agent list applies to every evaluation conversation unless a conversation's `agent` field selects a subset. Agent configurations can use any supported live driver, including `http_api` and `openshift_agentic_run`; see [OpenShift AgenticRun Evaluation](agentic_lightspeed_evaluation.md) for that driver’s setup.

## Evaluation data

Each test case supplies a query and expected answer. Each agent API generates an actual response; `custom:answer_correctness` compares it with the expected answer.

```yaml
- conversation_group_id: openshift_basics
  turns:
    - turn_id: define_openshift
      query: What is OpenShift?
      expected_response: OpenShift is Red Hat's Kubernetes-based hybrid cloud application platform.
      turn_metrics:
        - custom:answer_correctness
```

The expected answer stays fixed across agents and repetitions.

## Run

```bash
uv run lightspeed-eval \
  --system-config path/to/system.yaml \
  --eval-data path/to/eval_data.yaml \
  --output-dir ./eval_output
```

Use your own evaluation data and system configuration with reachable agent APIs. The regular CLI filters (`--tags`, `--conv-ids`, `--metrics`) still apply.

## Reports

Each invocation creates a timestamped `eval_<timestamp>/` directory under the output directory:

```text
eval_<timestamp>/
├── eval_report.json
├── baseline/
│   ├── agent_report.json
│   └── run_1/ ... run_N/
└── candidate/
    ├── agent_report.json
    └── run_1/ ... run_N/
```

Run directories contain normal evaluation artifacts. `agent_report.json` contains aggregated details for one agent; `eval_report.json` contains agent summaries and comparisons.

Reports include per-agent pass rates, per-metric summaries, latency/token totals, and pairwise deltas/rankings. With repeated per-case CSV results, reports also include `pass@k` overall and by metric/conversation.

Pairwise significance uses Fisher's exact test for aggregate PASS vs. non-PASS outcomes (FAIL, ERROR, and SKIPPED count as non-PASS) and Mann–Whitney U tests for per-run metric scores. Significance level is `0.05`; p-values are unadjusted across metrics. Tests may be omitted when sample counts are insufficient or scores are identical. Treat results as evidence from these runs, not proof of universal model quality.

## Run behavior

- Failed runs are recorded; other runs continue. Configuration or evaluation-data validation errors abort the evaluation.
- HTTP agent query caching is disabled for repetitions after first run; AgenticRun CR operations are not cached.
- `pass@k` uses PASS/FAIL case results; ERROR/SKIPPED cases do not contribute to its denominator.
- Parallel runs can overload APIs or host. Start sequentially; enable concurrency after checking capacity.
