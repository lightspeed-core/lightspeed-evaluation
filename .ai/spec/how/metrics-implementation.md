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
| `core/metrics/custom/loop_eval.py` | — | Deterministic consecutive tool-loop detection |
| `core/metrics/custom/openshift_agentic_run_eval.py` | `evaluate_openshift_agentic_run_status()` | OpenShift AgenticRun status assertions |
| `core/metrics/custom/mrr_eval.py` | `evaluate_mrr()` | MRR ranking with optional local embedding similarity |
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

**MetricManager** resolves metric lists and merges metadata; it does not register or dispatch handler classes. `MetricsEvaluator.handlers` maps framework prefixes to backend handlers. Each handler maintains its own `supported_metrics` mapping for metric names. New metrics must be registered in the appropriate handler and configured in system metadata; new backends also require evaluator dispatch.

**Metadata merging** starts with system defaults for the relevant evaluation level, then shallow-merges that level's override; override keys win while non-overlapping defaults remain. Turn-level and conversation-level metadata do not fall through to one another.

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

- **Ragas and DeepEval have different LLM wrapper contracts**: Ragas uses `llm_factory` with LiteLLM's async completion function; DeepEval uses its own model integration. `core/llm/ragas.py` and `core/llm/deepeval.py` bridge these interfaces.
- **LiteLLM patches** (`core/llm/litellm_patch.py`) exist because LiteLLM's provider routing sometimes needs corrections for specific model/provider combinations.
- **NLP metrics** load optional dependencies such as `sacrebleu`, `rouge-score`, and `rapidfuzz` only when the corresponding metric runs.
- **MRR** is implemented by `core/metrics/custom/mrr_eval.py`. `NLPMetrics` lazily loads Sentence Transformers using the `embedding_model` metric setting or `all-MiniLM-L6-v2`; unavailable dependencies or model-load failures use substring fallback.
- **Script metrics** execute external scripts via subprocess. The script manager runs the script file directly and checks the return code for pass/fail determination.
