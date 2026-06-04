# System Overview

LightSpeed Evaluation Framework is a CLI and programmatic tool for evaluating GenAI application responses. It takes YAML-defined evaluation datasets (conversations with turns), runs each turn through configurable metrics using LLM judges, and produces scored reports with statistical analysis. It supports both single-turn and multi-turn conversation evaluation, multiple evaluation backends, concurrent execution, and optional real-time data generation via pluggable agent drivers.

## Behavioral Rules

### System Role

1. The framework evaluates GenAI responses — it does not generate them (generation is handled by agent drivers or pre-populated data).
2. Evaluation is driven by two YAML configurations: a system config (judge LLMs, metrics, infrastructure) and evaluation data (conversations with turns to evaluate).
3. The CLI entry point is `lightspeed-eval`; the programmatic API is `lightspeed_evaluation.api`.

### Evaluation Levels

4. Turn-level evaluation scores individual query-response pairs within a conversation.
5. Conversation-level evaluation scores the entire multi-turn conversation as a whole.
6. Both levels can run in the same evaluation pass with different metric sets.

### Metric Resolution

7. When a conversation or turn specifies `metrics: null` (or omits metrics), the system uses default metrics from system config (metrics with `default: true`).
8. When a conversation or turn specifies `metrics: []` (empty list), evaluation is skipped for that level.
9. When a conversation or turn specifies explicit metrics, only those metrics run.
10. Metric metadata (thresholds, criteria, weights) follows a priority hierarchy: turn-level overrides > conversation-level overrides > system defaults.

### Execution Model

11. Conversations are evaluated concurrently using a configurable thread pool (`core.max_threads`).
12. Within a conversation, turns are evaluated sequentially in order.
13. If `skip_on_failure` is enabled, remaining turns in a conversation are skipped after the first FAIL or ERROR.
14. Agent API errors cascade ERROR to the current turn, all remaining turns, and conversation-level metrics — regardless of `skip_on_failure`. Setup script failures also cascade ERROR to all turns and conversation-level metrics (setup runs before any turns, so the entire conversation is marked).

### Multiple Expected Responses

15. When a turn has multiple `expected_response` values, each is evaluated sequentially.
16. Evaluation stops on the first PASS result.
17. If all fail, the highest-scoring result is returned.
18. Panel-of-judges mode does not support multiple expected responses (to avoid N judges x M responses complexity).

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `core.max_threads` | int | None (auto-scales to CPU count) | Thread pool size for concurrent conversation evaluation |
| `core.cache_enabled` | bool | true | Global cache toggle (AND-ed with component-level toggles) |
| `core.skip_on_failure` | bool | false | Skip remaining turns on first ERROR |
| `llm_pool` | list | — | Centralized pool of judge LLM configurations |
| `judge_panel` | object | — | Multi-judge panel with aggregation strategy |
| `embedding` | object | — | Embedding provider config for semantic metrics |
| `agents` | object | — | Agent driver configurations (replaces legacy `api` block) |
| `storage` | list | — | Storage backend configurations (file, sql) |
| `visualization` | object | — | Graph generation settings |
| `quality_score` | object | — | Metrics subset for composite quality scoring |

## Constraints

- Python 3.11+ required.
- `lsc_agent_eval/` is deprecated — new features go in `src/lightspeed_evaluation/`.
- `src/generate_answers/` is moving to a separate repo — no new features.
- Always use pytest-mock (`mocker.patch`), never `unittest.mock`.
- Do not disable lint warnings (`# noqa`, `# type: ignore`, `# pylint: disable`).
- All quality checks (`make pre-commit`) must pass before code is considered complete.

## Planned Changes

| Ticket | Summary |
|---|---|
| LEADS-337 | Multi-iteration evaluation (pass@k, variance for non-determinism) |
| LEADS-357 | OpenShift Agentic Lightspeed evaluation (event-driven agents) |
| LEADS-359 | Multi-agent framework (unified agent abstraction) |
| LEADS-372 | Skill safety & quality evaluation |
| LEADS-373 | Agentic skills standardization (regression detection, model comparison) |
