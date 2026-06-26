# Configuration and Models

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `core/models/system.py` | `SystemConfig` | Top-level system config Pydantic model |
| `core/models/data.py` | `EvaluationData`, `TurnData`, `MetricResult`, `EvaluationResult` | Evaluation dataset, turn, and result models |
| `core/models/agents.py` | `HttpApiAgentConfig`, `ProposalAgentConfig`, `AgentsConfig`, `AgentDefaultConfig` | Agent driver configuration models |
| `core/models/api.py` | Legacy API config | Backward-compatible API config (deprecated) |
| `core/models/llm.py` | LLM config models | LLM pool and judge panel models |
| `core/models/summary.py` | `EvaluationSummary` | Result aggregation models |
| `core/models/quality.py` | `QualityReport` | Quality score models |
| `core/models/statistics.py` | Statistical models | Bootstrap CI, distribution analysis |
| `core/models/mixins.py` | Shared behaviors | Pydantic model mixins |
| `core/system/loader.py` | `ConfigLoader` | YAML config loading and system config construction |
| `core/system/validator.py` | `DataValidator` | Evaluation data validation (metric prerequisites, turn structure) |

## Data Flow

1. CLI or API caller provides paths to two YAML files: system config and evaluation data.
2. `ConfigLoader` reads system config YAML → validates → produces `SystemConfig` Pydantic model.
3. `DataValidator` reads evaluation data YAML → validates → produces `EvaluationData` with `TurnData` entries.
4. During validation, metric data prerequisites are checked (e.g., RAG metrics need `contexts`).
5. `SystemConfig` performs auto-migration of legacy fields (e.g., `api:` → `agents:`).
6. Cache settings are unified: `core.cache_enabled` AND component `cache_enabled` must both be true.
7. GEval entries in metrics_metadata are validated at load time (fail-fast).

## Key Abstractions

**SystemConfig** is the single source of truth for all infrastructure configuration. It aggregates sub-configs for every system component. Auto-migration in validators ensures backward compatibility as config format evolves.

**EvaluationData** is a container of conversations. Each conversation contains ordered turns, optional scripts, and optional metric overrides. The three-mode metric resolution (null/empty/explicit) is encoded in the model's optional list fields.

**Pydantic v2 models** with `model_validator` decorators handle complex cross-field validation (e.g., quality_score metrics must exist in metrics_metadata). Google-style docstrings are enforced by pydocstyle.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `EvaluationPipeline` | `SystemConfig` | Reads all infrastructure settings |
| `EvaluationPipeline` | `EvaluationData` | Iterates conversations and turns |
| `LLMManager` | `SystemConfig.llm_pool` | Creates LLM instances from pool |
| `MetricManager` | `SystemConfig.default_*_metrics_metadata` | Resolves metric defaults |
| `StorageFactory` | `SystemConfig.storage` | Creates storage backends |

## Implementation Notes

- **Auto-migration is silent**: When `api:` is converted to `agents:`, no warning is logged. This is by design for frictionless upgrades but can surprise debugging.
- **Pydantic v2 validators** run in a specific order. Cross-field validators (`model_validator(mode='after')`) run after all field validators, which is critical for the cache unification logic.
- **The `mixins.py` file** contains shared behaviors that multiple models inherit — if you're adding a field pattern that appears in more than two models, check if a mixin already handles it.
- **Config loading does not merge files**: Each YAML file is loaded independently. There is no config inheritance or overlay mechanism between system config files.
