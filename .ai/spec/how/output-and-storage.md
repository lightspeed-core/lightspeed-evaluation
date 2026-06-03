# Output and Storage

## Module Map

| File | Key Symbols | Responsibility |
|---|---|---|
| `core/output/generator.py` | `OutputHandler` | Orchestrates report generation |
| `core/output/visualization.py` | — | Graph generation (matplotlib, seaborn) |
| `core/output/statistics.py` | — | Statistical computations (bootstrap CI, distributions) |
| `core/output/data_persistence.py` | — | File writing (CSV, JSON, TXT) |
| `core/storage/protocol.py` | `BaseStorageBackend` | Abstract storage interface |
| `core/storage/factory.py` | `create_pipeline_storage_backend()` | Backend instantiation |
| `core/storage/file_storage.py` | `FileStorageBackend` | File output + report generation |
| `core/storage/sql_storage.py` | `SQLStorageBackend` | Database persistence |
| `core/storage/composite_storage.py` | `CompositeStorageBackend` | Multi-backend chaining |
| `core/storage/config.py` | — | Storage configuration models |

## Data Flow

1. During evaluation, `EvaluationPipeline` calls `storage.save_run(results)` after each conversation completes.
2. After all conversations finish, `set_evaluation_context()` provides the full dataset, then `finalize()` is called.
3. `FileStorageBackend.save_run()` accumulates results in memory only — no disk writes. `finalize()` triggers `OutputHandler` to generate all reports from accumulated results.
4. `SQLStorageBackend.save_run()` commits results to the database immediately per conversation. `finalize()` is a no-op (logs a count).
5. `CompositeStorageBackend` delegates all calls to its child backends in order.

## Key Abstractions

**Storage lifecycle** is protocol-driven: `initialize()` → `save_run()` (repeated per conversation) → `set_evaluation_context()` → `finalize()` → `close()`. File and SQL backends implement this lifecycle differently: file storage defers all writes to `finalize()`, while SQL storage commits immediately in each `save_run()`.

**The factory pattern** in `create_pipeline_storage_backend()` reads the config's storage list and instantiates the appropriate backends. If multiple backends are configured, they're wrapped in a `CompositeStorageBackend`. When no storage is configured, a `NoOpStorageBackend` is returned.

**FileStorageBackend** accumulates results in memory during `save_run()` and needs `SystemConfig` plus the full evaluation dataset (`set_evaluation_context()`) to generate reports in `finalize()`. **SQLStorageBackend** commits to the database immediately per conversation and its `finalize()` is a no-op.

## Integration Points

| Consumer | Provider | Mechanism |
|---|---|---|
| `EvaluationPipeline` | `BaseStorageBackend` | Lifecycle calls (initialize, save_run, finalize, close) |
| `FileStorageBackend` | `OutputHandler` | Delegates report generation on finalize |
| `OutputHandler` | `EvaluationSummary` | Computes statistics for reports |
| `SQLStorageBackend` | SQLAlchemy | Database operations |

## Implementation Notes

- **SQLAlchemy URL encoding**: Connection strings with special characters (passwords, etc.) require URL encoding. The SQL backend handles this transparently.
- **NoOpStorageBackend** exists as a fallback when no storage is configured — it silently discards all data. This avoids null checks throughout the pipeline.
- **Graph generation** imports matplotlib and seaborn at call time, not at module level, because they're slow to import and not always needed.
- **Report paths**: Output files are written to the directory specified in config, with timestamped subdirectories per run.
- **File storage memory pressure**: Because file storage accumulates all results in memory until `finalize()`, very large evaluation runs may consume significant memory. SQL storage does not have this issue since it commits incrementally.
