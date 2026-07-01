# Metrics

Metrics are the scoring functions that evaluate LLM-powered application outputs (responses, context quality, tool calls, conversation flows, agentic workflow/proposal outcomes outcomes). The framework supports six metric backends, each with different capabilities, LLM requirements, and data prerequisites.

## Behavioral Rules

### Metric Categories

- **Ragas metrics** (`ragas:*`): Industry-standard RAG evaluation — faithfulness, answer relevancy, context precision/recall. Require LLM judge and often embedding model.
- **DeepEval metrics** (`deepeval:*`): Conversation-level analysis — conversation_completeness, conversation_relevancy, knowledge_retention. Require LLM judge.
- **User-defined criteria** (`geval:*`): LLM-as-judge with custom evaluation criteria defined in metric metadata (powered by DeepEval's GEval). Require LLM judge.
- **Custom metrics** (`custom:*`): Domain-specific evaluation registered under the custom framework. Some use LLM judges (answer_correctness, intent_eval, proposal_evaluation_correctness), others use pure comparison logic without LLM (keywords_eval, tool_eval, proposal_status).
- **NLP metrics** (`nlp:*`): No-LLM statistical metrics — BLEU, ROUGE, semantic similarity distance. No judge required.
- **Script metrics** (`script:*`): External Python scripts that perform real-world validation (e.g., OpenShift state checks). No judge required.

### Metric Resolution

- MetricManager resolves which metrics to run at each level (turn, conversation) using the three-mode system: null → defaults, empty list → skip, explicit list → use as-is.
- Metric metadata (thresholds, criteria, weights) is resolved per-metric by merging system defaults with level-specific overrides (turn or conversation). Override keys win, but non-overlapping system default keys are preserved. Turn-level metrics do not fall through to conversation-level metadata.
- Default metrics are those with `default: true` in system config's metrics_metadata.

### Scoring

- Every metric produces a score between 0.0 and 1.0.
- Status is determined by comparing score to threshold: score >= threshold → PASS, otherwise → FAIL.
- Default threshold is 0.5 when no threshold is configured for a metric.
- Metrics that error produce status ERROR with no score.

### Data Prerequisites

- RAG metrics (faithfulness, context_precision, etc.) require `contexts` in turn data.
- Comparison metrics require `expected_response` in turn data.
- Proposal assertion metrics require `expected_proposal_status` in turn data.
- Script metrics are skipped when the agent API is disabled (they need enriched turn data).
- Data prerequisite validation happens at two levels: `DataValidator` checks explicitly listed metrics against provided data at load time (fail-fast for obvious mismatches), and `MetricsEvaluator` rechecks during evaluation (catches default/resolved metrics and post-agent-enrichment data).

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
