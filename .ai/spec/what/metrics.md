# Metrics

Metrics are the scoring functions that evaluate GenAI responses. The framework supports six metric backends, each with different capabilities, LLM requirements, and data prerequisites.

## Behavioral Rules

### Metric Categories

1. **Ragas metrics** (`ragas:*`): Industry-standard RAG evaluation — faithfulness, answer relevancy, context precision/recall. Require LLM judge and often embedding model.
2. **DeepEval metrics** (`deepeval:*`): Conversation-level analysis — conversation_completeness, conversation_relevancy, knowledge_retention. Require LLM judge.
3. **GEval metrics** (`geval:*`): LLM-as-judge with custom evaluation criteria defined in metric metadata. Require LLM judge.
4. **Custom metrics** (`custom:*`): User-defined LLM prompt templates for domain-specific evaluation. Require LLM judge.
5. **NLP metrics** (`nlp:*`): No-LLM statistical metrics — BLEU, ROUGE, keyword matching, similarity distance. No judge required.
6. **Script metrics** (`script:*`): External Python scripts that perform real-world validation (e.g., Kubernetes state checks). No judge required.

### Metric Resolution

7. MetricManager resolves which metrics to run at each level (turn, conversation) using the three-mode system: null → defaults, empty list → skip, explicit list → use as-is.
8. Metric metadata (thresholds, criteria, weights) is resolved per-metric with the priority: turn-level override > conversation-level override > system default.
9. Default metrics are those with `default: true` in system config's metrics_metadata.

### Scoring

10. Every metric produces a score between 0.0 and 1.0.
11. Status is determined by comparing score to threshold: score >= threshold → PASS, otherwise → FAIL.
12. Default threshold is 0.5 when no threshold is configured for a metric.
13. Metrics that error produce status ERROR with no score.

### Data Prerequisites

14. RAG metrics (faithfulness, context_precision, etc.) require `contexts` in turn data.
15. Comparison metrics require `expected_response` in turn data.
16. Script metrics are skipped when the agent API is disabled (they need enriched turn data).
17. Data prerequisite validation happens at two levels: `DataValidator` checks explicitly listed metrics against provided data at load time (fail-fast for obvious mismatches), and `MetricsEvaluator` rechecks during evaluation (catches default/resolved metrics and post-agent-enrichment data).

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `default_turn_metrics_metadata` | dict | — | Per-metric config: threshold, criteria, model, strictness |
| `default_conversation_metrics_metadata` | dict | — | Same structure for conversation-level metrics |
| `quality_score.metrics` | list | — | Subset of metrics used for composite quality scoring |

## Constraints

- Metrics must be registered in MetricManager's supported_metrics dictionary to be usable.
- New custom metrics are added to `core/metrics/custom/` and registered in MetricManager.
- GEval entries are validated at config load time (fail-fast on invalid criteria).
- Panel-of-judges does not support multiple expected_response values per turn.
