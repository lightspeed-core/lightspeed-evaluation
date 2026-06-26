# Metrics Implementation

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `core/metrics/manager.py` | `MetricManager` | Metric resolution, registration, metadata merging |
| `core/metrics/ragas.py` | `RagasMetrics` | Ragas framework integration |
| `core/metrics/deepeval.py` | `DeepEvalMetrics` | DeepEval framework integration |
| `core/metrics/geval.py` | `GEvalHandler` | GEval custom-criteria evaluation |
| `core/metrics/nlp.py` | `NLPMetrics` | Statistical NLP metrics (no LLM) |
| `core/metrics/script.py` | `ScriptEvalMetrics` | External script execution |
| `core/metrics/custom/custom.py` | `CustomMetrics` | Custom LLM-based metric handler |
| `core/metrics/custom/keywords_eval.py` | — | Keyword matching evaluation logic |
| `core/metrics/custom/tool_eval.py` | — | Tool use evaluation logic |
| `core/metrics/custom/proposal_eval.py` | `evaluate_proposal_status()` | Proposal status assertion metric (phase, duration, attempts, conditions) |
| `core/metrics/custom/prompts.py` | — | Prompt templates for custom metrics |
| `pipeline/evaluation/evaluator.py` | `MetricsEvaluator` | Metric dispatch, multi-expected-response logic, status determination |
| `pipeline/evaluation/judges.py` | `JudgeOrchestrator` | Panel scoring, aggregation strategies |

## Data Flow

1. `ConversationProcessor` resolves which metrics to run via `MetricManager.resolve_metrics()`.
2. For each metric, `MetricsEvaluator.evaluate_metric()` is called.
3. The evaluator determines the metric's framework prefix (`ragas:`, `deepeval:`, `geval:`, `nlp:`, `custom:`, `script:`) and routes to the appropriate handler.
4. The handler calls the underlying framework (Ragas SDK, DeepEval SDK, or custom logic) with the appropriate LLM wrapper.
5. The handler returns a raw score.
6. The evaluator applies threshold logic: `score >= threshold` → PASS, else FAIL.
7. If panel-of-judges is active for this metric, `JudgeOrchestrator` runs the metric across all judges and aggregates.

## Key Abstractions

**MetricManager** uses a `supported_metrics` dictionary that maps metric name prefixes to handler classes. Adding a new metric backend means: (1) create a handler class in `core/metrics/`, (2) register it in `supported_metrics`, (3) add metadata to system config.

**Metadata merging** follows a three-layer cascade: system defaults → conversation-level overrides → turn-level overrides. Each layer can override threshold, criteria, model, strictness, and other metric-specific settings. The merge is shallow — a turn-level override replaces the entire metadata dict for that metric, not individual keys.

**Handler isolation**: Each metric backend (Ragas, DeepEval, etc.) manages its own LLM wrapper. Ragas uses `core/llm/ragas.py` wrappers, DeepEval uses `core/llm/deepeval.py` wrappers. This isolates framework-specific quirks from the generic pipeline.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `ConversationProcessor` | `MetricManager` | Calls `resolve_metrics()` for metric list; `MetricsEvaluator` calls `get_effective_threshold()` for thresholds and `get_metric_metadata()` for merged metric metadata |
| `MetricsEvaluator` | `RagasMetrics`, `DeepEvalMetrics`, etc. | Handler dispatch by metric prefix |
| `MetricsEvaluator` | `JudgeOrchestrator` | Delegates panel evaluation when panel is active |
| `JudgeOrchestrator` | Per-judge `LLMManager` | Each judge has its own LLMManager instance |
| `MetricsEvaluator` | `TokenTracker` | Extracts token counts from LLM responses after each evaluation |

## Implementation Notes

- **Ragas and DeepEval have different LLM wrapper contracts**: Ragas expects `langchain`-style interfaces; DeepEval expects its own model classes. The `core/llm/ragas.py` and `core/llm/deepeval.py` files bridge this gap.
- **LiteLLM patches** (`core/llm/litellm_patch.py`) exist because LiteLLM's provider routing sometimes needs corrections for specific model/provider combinations.
- **NLP metrics** import heavy dependencies (sacrebleu, rouge-score, rapidfuzz) via lazy imports — they're optional and only loaded when an `nlp:` metric is actually used.
- **Script metrics** execute external scripts via subprocess. The script manager runs the script file directly and checks the return code for pass/fail determination.
