# Output and Storage

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `core/output/generator.py` | `OutputHandler` | Orchestrates report generation (CSV, JSON, TXT, quality report) |
| `core/output/visualization.py` | — | Graph generation (matplotlib, seaborn): pass_rates, score_distribution, status_breakdown, conversation_heatmap |
| `core/output/statistics.py` | — | Statistical computations (bootstrap CI, distributions) |
| `core/output/data_persistence.py` | — | File writing (CSV, JSON, TXT) |
| `core/storage/protocol.py` | `BaseStorageBackend` | Abstract storage interface |
| `core/storage/factory.py` | `create_pipeline_storage_backend()` | Backend instantiation from config |
| `core/storage/file_storage.py` | `FileStorageBackend` | File output + report generation |
| `core/storage/sql_storage.py` | `SQLStorageBackend` | Database persistence |
| `core/storage/langfuse_storage.py` | `LangfuseStorageBackend` | Langfuse observability platform persistence |
| `core/storage/composite_storage.py` | `CompositeStorageBackend` | Multi-backend chaining |
| `core/storage/config.py` | `FileBackendConfig`, `DatabaseBackendConfig`, `LangfuseBackendConfig` | Storage configuration models |

## Data Flow

- During evaluation, `EvaluationPipeline` calls `storage.save_run(results)` after each conversation completes.
- After all conversations finish, `set_evaluation_context()` provides the full dataset, then `finalize()` is called.
- `FileStorageBackend.save_run()` accumulates results in memory only — no disk writes. `finalize()` triggers `OutputHandler` to generate all reports from accumulated results.
- `SQLStorageBackend.save_run()` commits results to the database immediately per conversation. `finalize()` is a no-op (logs a count).
- `LangfuseStorageBackend` accumulates results during `save_run()`, then creates a trace span and writes individual scores via `create_score()` on `finalize()`.
- `CompositeStorageBackend` delegates all calls to its child backends in order.

## Key Abstractions

**Storage lifecycle** is protocol-driven: `initialize()` → `save_run()` (repeated per conversation) → `set_evaluation_context()` → `finalize()` → `close()`. Each backend implements this differently: file defers writes, SQL commits incrementally, Langfuse accumulates then flushes.

**The factory pattern** in `create_pipeline_storage_backend()` reads the config's storage list and instantiates the appropriate backends (file, sql, langfuse). If multiple backends are configured, they're wrapped in a `CompositeStorageBackend`. When no storage is configured, a `NoOpStorageBackend` is returned.

**FileStorageBackend** accumulates results in memory during `save_run()` and needs `SystemConfig` plus the full evaluation dataset (`set_evaluation_context()`) to generate reports in `finalize()`. **SQLStorageBackend** commits to the database immediately per conversation and its `finalize()` is a no-op. **LangfuseStorageBackend** accumulates results and writes traces/scores to Langfuse on `finalize()`.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `EvaluationPipeline` | `BaseStorageBackend` | Lifecycle calls (initialize, save_run, finalize, close) |
| `FileStorageBackend` | `OutputHandler` | Delegates report generation on finalize |
| `OutputHandler` | `EvaluationSummary` | Computes statistics for reports |
| `SQLStorageBackend` | SQLAlchemy | Database operations |
| `LangfuseStorageBackend` | Langfuse SDK | Trace and score creation |

## Implementation Notes

- **SQLAlchemy URL encoding**: Connection strings with special characters (passwords, etc.) require URL encoding. The SQL backend handles this transparently.
- **NoOpStorageBackend** exists as a fallback when no storage is configured — it silently discards all data. This avoids null checks throughout the pipeline.
- **Graph generation** imports matplotlib and seaborn at call time, not at module level, because they're slow to import and not always needed.
- **Report paths**: Output files are written to the directory specified in config, with timestamped subdirectories per run.
- **File storage memory pressure**: Because file storage accumulates all results in memory until `finalize()`, very large evaluation runs may consume significant memory. SQL storage does not have this issue since it commits incrementally.
- **Langfuse** requires the `langfuse` optional dependency (>=4.0.0). Config supports inline credentials or environment variable fallback.
