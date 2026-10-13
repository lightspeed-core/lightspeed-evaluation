# Metrics

Metrics are the scoring functions that evaluate LLM-powered application outputs (responses, context quality, tool calls, conversation flows, and OpenShift AgenticRun outcomes). The framework supports six metric backends, each with different capabilities, LLM requirements, and data prerequisites.

## Behavioral Rules

### Metric Categories

- **Ragas metrics** (`ragas:*`): RAG evaluation — faithfulness, response_relevancy, context_relevance, context_precision, context_recall, context_utilization. Require LLM judge and often embedding model.
- **DeepEval metrics** (`deepeval:*`): Conversation-level analysis — conversation_completeness, conversation_relevancy, knowledge_retention. Require LLM judge.
- **User-defined criteria** (`geval:*`): LLM-as-judge with custom evaluation criteria defined in metric metadata (powered by DeepEval's GEval). Require LLM judge.
- **Custom metrics** (`custom:*`): Domain-specific evaluation registered under the custom framework. Some use LLM judges (answer_correctness, intent_eval, openshift_agentic_run_evaluation_correctness), others use deterministic checks (keywords_eval, tool_eval, loop_eval, openshift_agentic_run_status).
- **NLP metrics** (`nlp:*`): No-LLM metrics — BLEU, ROUGE, MRR, and semantic similarity distance. `nlp:mrr` is turn-level and requires `contexts` and `expected_contexts`; first relevant context at rank `r` scores `1/r`, and no match scores `0`. It uses Sentence Transformers for semantic matching when the model loads, otherwise normalized substring matching. Metric PASS/FAIL threshold defaults to `0.5`. No judge required.
- **Script metrics** (`script:*`): External Python scripts that perform real-world validation (e.g., OpenShift state checks). No judge required.

### Metric Resolution

- MetricManager resolves which metrics to run at each level (turn, conversation) using the three-mode system: null → defaults, empty list → skip, explicit list → use as-is.
- Metric metadata (thresholds, criteria, weights) is resolved per-metric by merging system defaults with level-specific overrides (turn or conversation). Override keys win, but non-overlapping system default keys are preserved. Turn-level metrics do not fall through to conversation-level metadata.
- Default metrics are those with `default: true` in system config's metrics_metadata.

### Scoring

- Successful numeric metrics produce scores between 0.0 and 1.0; unavailable inputs or evaluation errors can produce no score.
- Status is determined by comparing score to threshold: score >= threshold → PASS, otherwise → FAIL.
- Default threshold is 0.5 when no threshold is configured for a metric.
- Metrics that error produce status ERROR with no score.

### Data Prerequisites

- Ragas context metrics (faithfulness, context precision/recall, context relevance/utilization) require `contexts`; `ragas:response_relevancy` requires `query` and `response` and uses embeddings.
- Comparison metrics require `expected_response` in turn data.
- MRR requires both `contexts` and `expected_contexts`.
- OpenShift AgenticRun status assertions require `expected_openshift_agentic_run_status` in turn data. They can check run phase, maximum duration, conditions, and analysis, execution, or verification results; score is `1.0` if all configured checks pass and `0.0` on the first failure.
- `custom:openshift_agentic_run_evaluation_correctness` scores diagnosis, execution, and verification with an LLM judge. It scores only applicable phases; execution failures caused by infrastructure can be marked not applicable rather than treated as agent reasoning failures.
- Script metrics are not evaluated when no `agents` configuration exists or `agents.enabled` is false.
- Data prerequisite validation happens at two levels: `DataValidator` checks explicitly listed metrics against provided data at load time (fail-fast for obvious mismatches), and `MetricsEvaluator` rechecks during evaluation (catches default/resolved metrics and post-agent-enrichment data). If required inputs are unavailable during evaluation, the metric result is `ERROR` with no score; threshold comparison does not run, so this is not a threshold-based `FAIL`.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `default_turn_metrics_metadata` | dict | — | Per-metric config: threshold, criteria, model, strictness |
| `default_conversation_metrics_metadata` | dict | — | Same structure for conversation-level metrics |
| `quality_score.metrics` | list | — | Subset of metrics used for composite quality scoring |

## Constraints

- Each metric handler class (RagasMetrics, DeepEvalMetrics, CustomMetrics, NLPMetrics) maintains its own `supported_metrics` dictionary. MetricsEvaluator (not MetricManager) dispatches by metric prefix to the appropriate handler.
- New custom metrics are added to `core/metrics/custom/` and registered in the `CustomMetrics.supported_metrics` dictionary.
- GEval entries are validated at config load time (fail-fast on invalid criteria).
- Panel-of-judges does not support multiple expected_response values per turn.
- DeepEval and GEval may return `score=None` on transient failures; None scores are treated as ERROR status directly (no retry).
