# LLM and Judges

LLM judges score evaluation metrics. The framework supports single-judge and panel-of-judges modes with multiple LLM providers. LLM configuration has migrated from a single `llm:` block to a pool-based `llm_pool` + `judge_panel` model.

## Behavioral Rules

### LLM Providers

1. Supported providers: OpenAI, Azure, Anthropic, Gemini, Vertex AI, WatsonX, Ollama, hosted_vLLM.
2. Each provider requires specific environment variables (e.g., `OPENAI_API_KEY` for OpenAI).
3. Provider validation happens at LLMManager initialization — missing env vars fail fast.

### LLM Pool

4. `llm_pool` defines a centralized list of LLM configurations, each with a unique `id`.
5. Each pool entry specifies provider, model, and optional overrides (temperature, max_tokens, etc.).
6. The pool has a `defaults.cache_enabled` setting that AND-s with the global `core.cache_enabled` toggle. This applies to all pool entries uniformly (not per-entry).

### Judge Panel

7. `judge_panel` references models from `llm_pool` by ID.
8. When a panel is configured, each metric is evaluated by every judge independently.
9. Scores are aggregated using the configured strategy: `max`, `average`, or `majority_vote`.
10. `majority_vote` determines PASS/FAIL by majority, then uses the mean score across all valid judges (not just the majority group).
11. Each judge gets a separate cache directory (`judge_0`, `judge_1`, etc.) to isolate cached results.
12. The panel can be limited to specific metrics via `enabled_metrics`; unlisted metrics use a single judge.

### Single Judge (Legacy)

13. When no panel is configured, a single LLMManager serves as the judge for all metrics.
14. The legacy `llm:` config block is still supported but deprecated in favor of `llm_pool` + `judge_panel`.

### Token Tracking

15. Token usage (input/output) is tracked per-judge via direct extraction from LLM responses.
16. Token counts are recorded per metric result for cost analysis.
17. Total token usage is summarized in the final report.

### Embedding

18. Embedding models are used by Ragas metrics for semantic similarity evaluation.
19. Supported embedding providers: OpenAI, Gemini, HuggingFace (local).
20. HuggingFace embeddings require the `local-embeddings` optional dependency (includes torch).

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `llm_pool[].id` | string | — | Unique identifier for the LLM configuration |
| `llm_pool[].provider` | string | — | LLM provider name |
| `llm_pool[].model` | string | — | Model identifier |
| `llm_pool.defaults.cache_enabled` | bool | true | Pool-wide cache toggle (AND-ed with global) |
| `judge_panel.models` | list | — | List of llm_pool IDs to use as judges |
| `judge_panel.aggregation_strategy` | string | max | Score aggregation: max, average, majority_vote |
| `judge_panel.enabled_metrics` | list | null | Metrics to apply panel to (null = all) |
| `embedding.provider` | string | — | Embedding provider |
| `embedding.model` | string | — | Embedding model name |

## Constraints

- LLMManager can nest other LLMManagers (for judges) but nested managers lack system_config to prevent infinite recursion.
- Cache unification: `core.cache_enabled` AND component-level `cache_enabled` must both be true for caching to activate.
- Auto-migration: legacy `api:` block is automatically converted to `agents:` when agents config is absent.
