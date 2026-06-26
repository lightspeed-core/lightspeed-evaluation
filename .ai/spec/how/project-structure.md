# Project Structure

## Module Map

| File/Directory | Key Symbols | Responsibility |
|---|---|---|
| `src/lightspeed_evaluation/` | — | Package root |
| `api.py` | `evaluate()`, `evaluate_with_summary()`, `evaluate_conversation()`, `evaluate_turn()` | Programmatic API — clean entry points for Python callers |
| `runner/evaluation.py` | `main()` | CLI entry point (`lightspeed-eval` command) |
| `pipeline/evaluation/pipeline.py` | `EvaluationPipeline` | Top-level orchestrator — creates components, dispatches conversations |
| `pipeline/evaluation/processor.py` | `ConversationProcessor`, `ProcessorComponents` | Per-conversation processing — turn iteration, agent invocation, metric dispatch |
| `pipeline/evaluation/evaluator.py` | `MetricsEvaluator` | Metric evaluation engine — routes to framework handlers, handles multi-expected-response |
| `pipeline/evaluation/judges.py` | `JudgeOrchestrator` | Multi-judge coordination — aggregation strategies (max, average, majority_vote) |
| `pipeline/evaluation/driver.py` | `AgentDriver`, `HttpApiDriver`, `ProposalDriver`, `TerminalOutcome` | Agent driver abstraction, HTTP and Proposal implementations |
| `pipeline/evaluation/registry.py` | `AgentDriverRegistry`, `AGENT_DRIVERS` | Driver type registry and factory |
| `pipeline/evaluation/amender.py` | `APIDataAmender` | Mutates turn data with HTTP agent response, tokens, latency |
| `pipeline/evaluation/proposal_amender.py` | `ProposalAmender` | Fetches child Result CRs, builds Markdown summary for proposal turns |
| `pipeline/evaluation/cli.py` | `CLIClient`, `KubeCLI` | Abstract CLI abstraction and Kubernetes (oc/kubectl) implementation |
| `pipeline/evaluation/errors.py` | `EvaluationErrorHandler` | Pipeline error handling (marks metrics as ERROR on failures) |
| `core/metrics/manager.py` | `MetricManager` | Metric resolution (defaults vs overrides) and registration |
| `core/metrics/ragas.py` | `RagasMetrics` | Ragas framework adapter |
| `core/metrics/deepeval.py` | `DeepEvalMetrics` | DeepEval framework adapter |
| `core/metrics/geval.py` | `GEvalHandler` | GEval (LLM-as-judge with custom criteria) adapter |
| `core/metrics/nlp.py` | `NLPMetrics` | No-LLM statistical metrics (BLEU, ROUGE, similarity) |
| `core/metrics/script.py` | `ScriptEvalMetrics` | External script execution for validation |
| `core/metrics/custom/custom.py` | `CustomMetrics` | Custom LLM-based metric handler |
| `core/metrics/custom/keywords_eval.py` | — | Keyword matching evaluation |
| `core/metrics/custom/tool_eval.py` | — | Tool use evaluation |
| `core/metrics/custom/proposal_eval.py` | `evaluate_proposal_status()` | Proposal status assertion metric |
| `core/metrics/custom/prompts.py` | — | Prompt templates for custom metrics |
| `core/llm/manager.py` | `LLMManager` | LLM provider abstraction, judge panel creation |
| `core/llm/token_tracker.py` | `TokenTracker` | Token usage accounting |
| `core/llm/ragas.py` | — | Ragas-specific LLM wrappers |
| `core/llm/deepeval.py` | — | DeepEval-specific LLM wrappers |
| `core/llm/custom.py` | — | Custom metric LLM interaction |
| `core/llm/litellm_patch.py` | — | LiteLLM compatibility patches |
| `core/embedding/manager.py` | `EmbeddingManager` | Embedding provider abstraction (lazy validation via `ensure_ready()`) |
| `core/embedding/ragas.py` | — | Ragas-specific embedding wrappers |
| `core/models/system.py` | `SystemConfig` | Top-level system configuration model |
| `core/models/data.py` | `EvaluationData`, `TurnData`, `MetricResult`, `EvaluationResult`, `EvaluationRequest`, `ConversationMetadata`, `DatasetMetadata` | Evaluation dataset, turn, result, and metadata models |
| `core/models/agents.py` | `HttpApiAgentConfig`, `ProposalAgentConfig`, `AgentsConfig`, `AgentDefaultConfig` | Agent driver configuration models |
| `core/models/api.py` | — | Legacy API configuration (deprecated) |
| `core/models/summary.py` | `EvaluationSummary` | Evaluation result aggregation |
| `core/models/quality.py` | `QualityReport` | Quality score computation |
| `core/models/statistics.py` | — | Statistical computation models |
| `core/models/llm.py` | — | LLM configuration models |
| `core/models/mixins.py` | — | Shared Pydantic model behaviors |
| `core/output/generator.py` | `OutputHandler` | Report generation orchestrator (CSV, JSON, TXT, quality report) |
| `core/output/visualization.py` | — | Graph generation (matplotlib/seaborn) |
| `core/output/statistics.py` | — | Statistical analysis computation |
| `core/output/data_persistence.py` | — | File-based output writing |
| `core/output/serializers.py` | — | Result serialization helpers |
| `core/storage/protocol.py` | `BaseStorageBackend` | Storage backend interface |
| `core/storage/factory.py` | `create_pipeline_storage_backend()` | Backend instantiation from config |
| `core/storage/sql_storage.py` | `SQLStorageBackend` | Database persistence |
| `core/storage/file_storage.py` | `FileStorageBackend` | File-based persistence + reports |
| `core/storage/langfuse_storage.py` | `LangfuseStorageBackend` | Langfuse observability platform persistence |
| `core/storage/composite_storage.py` | `CompositeStorageBackend` | Multi-backend chaining |
| `core/storage/config.py` | `FileBackendConfig`, `DatabaseBackendConfig`, `LangfuseBackendConfig` | Storage configuration models |
| `core/api/client.py` | `APIClient` | HTTP client with caching, retries; supports query/streaming/infer/responses |
| `core/api/streaming_parser.py` | `parse_streaming_response()`, `StreamingContext` | SSE parsing with TTFT/throughput tracking |
| `core/proposal/phase.py` | `derive_phase()` | Proposal phase derivation from CRD conditions |
| `core/system/validator.py` | `DataValidator` | Data validation (metric prerequisites, turn structure) |
| `core/system/loader.py` | `ConfigLoader` | YAML config loading and system config construction |
| `core/system/exceptions.py` | `ConfigurationError`, `EvaluationError`, etc. | Custom exception hierarchy |
| `core/system/env_validator.py` | `validate_openai_env()`, etc. | Per-provider environment variable validation |
| `core/system/setup.py` | — | Environment setup utilities |
| `core/system/ssl_certifi.py` | — | SSL certificate handling |
| `core/system/lazy_import.py` | — | Deferred imports for optional deps |
| `core/script/manager.py` | `ScriptExecutionManager` | Setup/cleanup script execution |
| `core/constants.py` | — | Framework-wide constants (thresholds, supported types, graph types) |

## Key Entry Points

- **CLI**: `runner/evaluation.py:main()` → parses YAML configs → `EvaluationPipeline.run_evaluation()`
- **Programmatic**: `api.py:evaluate()` → same pipeline, returns raw results
- **Programmatic with stats**: `api.py:evaluate_with_summary()` → pipeline + EvaluationSummary

## Naming Conventions

- Metric adapters are named by their backend: `ragas.py`, `deepeval.py`, `geval.py`, `nlp.py`, `script.py`
- Each `core/` subdirectory has an `__init__.py` that re-exports key symbols
- Models use Pydantic v2 with Google-style docstrings
- Test files mirror source structure: `tests/core/metrics/test_manager.py` tests `core/metrics/manager.py`
