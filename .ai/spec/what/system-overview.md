# System Overview

Lightspeed Evaluation Framework is a CLI and programmatic tool for evaluating LLM-powered application outputs — responses, context quality, tool calls, conversation flows, and agentic workflow (proposal) outcomes — in both live and offline modes. It takes YAML-defined evaluation datasets (conversations with turns), runs each turn through configurable metrics using LLM judges, and produces scored reports with statistical analysis. It supports multiple evaluation backends (Ragas, DeepEval, NLP, custom, script-based), user-defined evaluation criteria, panel-of-judges scoring, environment setup/cleanup scripts, pluggable agent drivers, concurrent execution, and both single-turn and multi-turn conversation evaluation.

## Behavioral Rules

### System Role

- The framework operates in two modes: **live mode** where agent drivers collect responses from external services and then evaluate them, and **offline mode** where pre-populated responses are evaluated directly.
- Evaluation is driven by two YAML configurations: a system config (judge LLMs, metrics, infrastructure) and evaluation data (conversations with turns to evaluate).
- The CLI entry point is `lightspeed-eval`; the programmatic API is `lightspeed_evaluation.api`.

### Evaluation Levels

- Turn-level evaluation scores individual query-response pairs within a conversation.
- Conversation-level evaluation scores the entire multi-turn conversation as a whole.
- Both levels can run in the same evaluation pass with different metric sets.

### Metric Resolution

- When a conversation or turn specifies `metrics: null` (or omits metrics), the system uses default metrics from system config (metrics with `default: true`).
- When a conversation or turn specifies `metrics: []` (empty list), evaluation is skipped for that level.
- When a conversation or turn specifies explicit metrics, only those metrics run.
- Metric metadata (thresholds, criteria, weights) is resolved by merging system defaults with level-specific overrides (turn or conversation). Override keys win, but non-overlapping system default keys are preserved. Turn-level metrics do not fall through to conversation-level metadata.

### Execution Model

- Conversations are evaluated concurrently using a configurable thread pool (`core.max_threads`).
- Within a conversation, turns are evaluated sequentially in order.
- If `skip_on_failure` is enabled, remaining turns and conversation-level metrics in a conversation are skipped after the first FAIL or ERROR.
- Agent API errors cascade ERROR to the current turn, all remaining turns, and conversation-level metrics — regardless of `skip_on_failure`. Setup script failures also cascade ERROR to all turns and conversation-level metrics (setup runs before any turns, so the entire conversation is marked).

### Multiple Expected Responses

- When a turn has multiple `expected_response` values, each is evaluated sequentially.
- Evaluation stops on the first PASS result.
- If all fail, the highest-scoring result is returned.
- Panel-of-judges mode does not support multiple expected responses (to avoid N judges x M responses complexity).

### User-Defined Metadata

- Each conversation can carry a `ConversationMetadata` object with standard fields (scenario_category, use_case, interaction_type, topic, complexity, data_source, persona, etc.) plus an `additional_metadata` dict for custom key-value pairs.
- The evaluation data can carry a `DatasetMetadata` object with dataset-level fields (description, team_product, dataset_version, generation_tools, llms_used, etc.) plus an `additional_metadata` dict.
- Metadata stays at the conversation/dataset level for traceability and quality grading — it is not carried into individual EvaluationResult rows.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `core.max_threads` | int | None (auto-scales to min(32, cpu_count+4)) | Thread pool size for concurrent conversation evaluation |
| `core.cache_enabled` | bool | true | Global cache toggle (AND-ed with component-level toggles) |
| `core.cache_base_dir` | string | .caches | Base directory for all cache storage |
| `core.skip_on_failure` | bool | false | Skip remaining turns and conversation metrics on first FAIL or ERROR |
| `core.fail_on_invalid_data` | bool | true | Fail evaluation on invalid data (vs. skip with warning) |
| `llm_pool` | list | — | Centralized pool of judge LLM configurations |
| `judge_panel` | object | — | Multi-judge panel with aggregation strategy |
| `embedding` | object | — | Embedding provider config for semantic metrics |
| `agents` | object | — | Agent driver configurations (replaces legacy `api` block) |
| `storage` | list | — | Storage backend configurations (file, sql, langfuse) |
| `visualization` | object | — | Graph generation settings |
| `logging` | object | — | Logging configuration (source_level, package_level, log_format, show_timestamps) |
| `quality_score` | object | — | Metrics subset for composite quality scoring |

## Constraints

- Python 3.11+ required.
- `lsc_agent_eval/` is deprecated — new features go in `src/lightspeed_evaluation/`.
- `src/generate_answers/` is moving to a separate repo — no new features.
- Always use pytest-mock (`mocker.patch`), never `unittest.mock`.
- Do not disable lint warnings (`# noqa`, `# type: ignore`, `# pylint: disable`).
- All quality checks (`make pre-commit`) must pass before code is considered complete.
