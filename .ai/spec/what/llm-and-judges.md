# LLM and Judges

LLM judges score evaluation metrics. The framework supports single-judge and panel-of-judges modes with multiple LLM providers. LLM configuration has migrated from a single `llm:` block to a pool-based `llm_pool` + `judge_panel` model.

## Behavioral Rules

### LLM Providers

- Supported providers: OpenAI, Azure, Anthropic, Gemini, Vertex AI, WatsonX, Ollama, hosted_vLLM.
- Each provider requires specific environment variables (e.g., `OPENAI_API_KEY` for OpenAI).
- Provider validation happens at LLMManager initialization — missing env vars fail fast.

### LLM Pool

- `llm_pool` defines a centralized list of LLM configurations, each with a unique `id`.
- Each pool entry specifies provider, model, and optional overrides (temperature, max_tokens, etc.).
- The pool has a `defaults.cache_enabled` setting that AND-s with the global `core.cache_enabled` toggle. This applies to all pool entries uniformly (not per-entry).

### Judge Panel

- `judge_panel` references models from `llm_pool` by ID.
- When a panel is configured, each metric is evaluated by every judge independently.
- Scores are aggregated using the configured strategy: `max`, `average`, or `majority_vote`.
- `majority_vote` determines PASS/FAIL by majority, then uses the mean score across all valid judges (not just the majority group).
- Each judge gets a separate cache directory (`judge_0`, `judge_1`, etc.) to isolate cached results.
- The panel can be limited to specific metrics via `enabled_metrics`; unlisted metrics use a single judge.

### Single Judge (Legacy)

- When no panel is configured, a single LLMManager serves as the judge for all metrics.
- The legacy `llm:` config block is still supported but deprecated in favor of `llm_pool` + `judge_panel`.

### Token Tracking

- Token usage (input/output) is tracked per-judge via a litellm hook that feeds into TokenTracker.
- Token counts are recorded per metric result for cost analysis (JudgeScore captures per-judge breakdown).
- Total token usage is summarized in the final report.

### Embedding

- Embedding models are used by Ragas metrics for semantic similarity evaluation.
- Supported embedding providers: OpenAI, Gemini, HuggingFace (local).
- HuggingFace embeddings require the `local-embeddings` optional dependency (includes torch).
- Embedding validation is lazy — `EmbeddingManager.ensure_ready()` defers provider/env validation until first use, so missing embedding credentials don't fail pipeline startup when embeddings aren't needed.

## Configuration Surface

| Field/Flag | Type | Default | Description |
|---|---|---|---|
| `llm_pool.models.<id>` | dict key | — | Unique identifier for the LLM configuration (dict key, not a field) |
| `llm_pool.models.<id>.provider` | string | — | LLM provider name |
| `llm_pool.models.<id>.model` | string | — | Model identifier |
| `llm_pool.defaults.cache_enabled` | bool | true | Pool-wide cache toggle (AND-ed with global) |
| `judge_panel.judges` | list | — | List of llm_pool IDs to use as judges |
| `judge_panel.aggregation_strategy` | string | max | Score aggregation: max, average, majority_vote |
| `judge_panel.enabled_metrics` | list | null | Metrics to apply panel to (null = all) |
| `embedding.provider` | string | — | Embedding provider |
| `embedding.model` | string | — | Embedding model name |

## Constraints

- LLMManager can nest other LLMManagers (for judges) but nested managers lack system_config to prevent infinite recursion.
- Cache unification: `core.cache_enabled` AND component-level `cache_enabled` must both be true for caching to activate.
- Auto-migration: legacy `api:` block is automatically converted to `agents:` when agents config is absent.
