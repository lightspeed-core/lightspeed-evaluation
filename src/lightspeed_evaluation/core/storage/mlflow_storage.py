"""MLflow storage backend for evaluation results.

Implements :class:`~lightspeed_evaluation.core.storage.protocol.BaseStorageBackend`
so MLflow plugs into the standard pipeline storage lifecycle without any
changes to the runner, API, or pipeline modules.

Install with: ``pip install 'lightspeed-evaluation[mlflow]'``

Requires **MLflow** (``mlflow>=2.14.0``) for tracing support via ``start_span()``.

Credentials are resolved from :class:`MLflowBackendConfig` fields first,
then from ``MLFLOW_TRACKING_URI`` environment variable as fallback (standard
MLflow behavior).

Lifecycle:
    1. ``initialize(run_info)`` — sets the experiment, starts an MLflow run,
       and logs run-level params (including model/judge/dataset context).
    2. ``save_result(result)``  — logs metrics for a single result immediately
       and accumulates data for per-metric aggregates and the results table.
    3. ``save_run(results)``    — logs metrics for each result in the batch
       immediately (called per conversation).
    4. ``finalize(success=...)`` — registers models (``provider>model``),
       logs per-metric aggregate metrics, a results table artifact, tags
       ``eval_status``, and ends the run as ``FINISHED`` (complete) or
       ``FAILED`` (aborted).
    5. ``close()``              — ends the run as ``FAILED`` if still active
       (finalize was skipped).
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from lightspeed_evaluation.core.models.data import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.storage.config import MLflowBackendConfig
from lightspeed_evaluation.core.storage.protocol import RunInfo
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

if TYPE_CHECKING:
    from lightspeed_evaluation.core.models.system import SystemConfig

logger = logging.getLogger(__name__)

_HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None

_MLFLOW_ERRORS: tuple[type[Exception], ...] = (
    RuntimeError,
    ValueError,
    OSError,
    ConnectionError,
)
try:
    _mlflow_exc = importlib.import_module("mlflow.exceptions")
    _MLFLOW_ERRORS = (*_MLFLOW_ERRORS, _mlflow_exc.MlflowException)
except (ImportError, ModuleNotFoundError):
    pass


@dataclass
class _PerMetricAccumulator:
    """Scores and pass/fail values for a single metric identifier."""

    scores: list[float] = field(default_factory=list)
    pass_values: list[float] = field(default_factory=list)


@dataclass
class _RunAccumulators:
    """Accumulated values for computing aggregate metrics at finalize.

    Attributes:
        step: Sequential result index (used as MLflow step).
        scores: Collected numeric scores for mean computation.
        pass_values: 1.0/0.0 per result for pass rate.
        latencies: Evaluation latencies for mean computation.
        token_totals: Running totals keyed by token category name.
        per_metric: Per-metric scores and pass values for comparison.
        table_rows: Accumulated result dicts for ``log_table()``.
    """

    step: int = 0
    scores: list[float] = field(default_factory=list)
    pass_values: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    token_totals: dict[str, float] = field(default_factory=dict)
    per_metric: dict[str, _PerMetricAccumulator] = field(default_factory=dict)
    table_rows: list[dict[str, Any]] = field(default_factory=list)


class MLflowStorageBackend:
    """Storage backend that exports evaluation results to MLflow incrementally.

    Creates one MLflow run per evaluation run. Each evaluation result is
    logged immediately as it arrives (via ``save_result`` or ``save_run``),
    using the MLflow ``step`` parameter as a sequential index. Per-metric
    aggregate metrics and a results table artifact are logged at
    ``finalize()``.

    Complete vs failed evals are distinguished at finalize time:
    ``success=True`` ends the MLflow run as ``FINISHED`` with tag
    ``eval_status=complete``; ``success=False`` ends as ``FAILED`` with
    ``eval_status=failed``. Incremental metrics remain available in both
    cases; aggregates are still written so partial runs stay inspectable.

    Judge, embedding, and agent models from system config are registered in
    the MLflow Model Registry as ``provider>model`` (``>`` replaces ``/``
    because MLflow disallows ``/`` and ``:`` in registered model names).
    Each finalize creates a new model version linked to the eval run so they
    appear under **Registered models** in the run UI.

    Results with ``score=None`` (ERROR/SKIPPED) are skipped from numeric
    scoring but their pass/fail status is still logged.

    All MLflow SDK errors are caught and logged — they never fail
    the evaluation pipeline.
    """

    def __init__(
        self,
        config: MLflowBackendConfig,
        system_config: Optional[SystemConfig] = None,
    ) -> None:
        """Initialize the MLflow storage backend.

        Args:
            config: MLflow backend configuration with optional tracking_uri,
                experiment_name fields.
            system_config: Optional system configuration for logging run
                params (model, judge panel, dataset context).
        """
        self._config = config
        self._system_config = system_config
        self._run: Any = None
        self._client: Any = None
        self._run_info: Optional[RunInfo] = None
        self._acc = _RunAccumulators()

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        return "mlflow"

    @property
    def results_count(self) -> int:
        """Return the number of results saved in this run."""
        return self._acc.step

    def initialize(self, run_info: RunInfo) -> None:
        """Set the MLflow experiment and start a new run.

        Args:
            run_info: Information about the evaluation run.
        """
        self._run_info = run_info
        self._acc = _RunAccumulators()

        if not _HAS_MLFLOW:
            logger.error(
                "mlflow is not installed. "
                "Add: pip install 'lightspeed-evaluation[mlflow]'"
            )
            return

        mlflow_mod = importlib.import_module("mlflow")

        try:
            if self._config.tracking_uri:
                mlflow_mod.set_tracking_uri(self._config.tracking_uri)

            mlflow_mod.set_experiment(self._config.experiment_name)

            run_name = run_info.name or f"eval_{run_info.run_id[:8]}"
            self._run = mlflow_mod.start_run(run_name=run_name)
            self._client = mlflow_mod

            self._log_run_params(run_name, run_info)
            self._client.set_tag("eval_status", "running")
            logger.info("MLflow backend initialized (run_id=%s)", run_info.run_id)
        except _MLFLOW_ERRORS:
            logger.exception("mlflow: failed to initialize run")
            self._run = None
            self._client = None

    def save_result(self, result: EvaluationResult) -> None:
        """Log metrics and a trace for a single evaluation result immediately.

        Args:
            result: The evaluation result to log.
        """
        if self._client is None or self._run is None:
            return

        try:
            self._log_result_metrics(result)
            self._log_result_trace(result)
        except _MLFLOW_ERRORS:
            logger.exception("mlflow: failed to log result at step %d", self._acc.step)

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Log metrics and traces for a batch of results immediately.

        Args:
            results: List of evaluation results to log.
        """
        if self._client is None or self._run is None:
            return

        for result in results:
            try:
                self._log_result_metrics(result)
                self._log_result_trace(result)
            except _MLFLOW_ERRORS:
                logger.exception(
                    "mlflow: failed to log result at step %d", self._acc.step
                )

    def set_evaluation_context(
        self, evaluation_data: Optional[list[EvaluationData]] = None
    ) -> None:
        """No-op — MLflow export does not need the full evaluation dataset."""
        _ = evaluation_data

    def finalize(self, success: bool = True) -> None:
        """Log aggregates, tag eval status, and end the MLflow run.

        Args:
            success: ``True`` for a complete eval (``FINISHED`` /
                ``eval_status=complete``); ``False`` for an aborted eval
                (``FAILED`` / ``eval_status=failed``). Incremental metrics
                and final aggregates are logged in both cases.
        """
        if self._client is None or self._run is None:
            return

        status = "FINISHED" if success else "FAILED"
        eval_status = "complete" if success else "failed"
        try:
            self._client.set_tag("eval_status", eval_status)
            self._register_models()
            self._log_aggregate_metrics()
            self._log_results_table()
        except _MLFLOW_ERRORS:
            logger.exception("mlflow: failed to log aggregate metrics or table")
        finally:
            self._end_run(status=status)

    def close(self) -> None:
        """End the MLflow run as failed if finalize was not called."""
        self._end_run(status="FAILED")

    def _end_run(self, status: str = "FINISHED") -> None:
        """Safely end the active MLflow run with the given status.

        Args:
            status: MLflow run status — typically ``FINISHED`` or ``FAILED``.
        """
        if self._client is not None and self._run is not None:
            try:
                if status == "FAILED":
                    try:
                        self._client.set_tag("eval_status", "failed")
                    except _MLFLOW_ERRORS:
                        logger.debug("mlflow: set_tag(eval_status) raised; ignoring")
                self._client.end_run(status=status)
            except _MLFLOW_ERRORS:
                logger.debug("mlflow: end_run raised; ignoring")
        self._run = None

    def _log_run_params(self, run_name: str, run_info: RunInfo) -> None:
        """Log run-level params including model and judge configuration.

        Args:
            run_name: Human-readable name for the run.
            run_info: Run information with ID and timestamp.
        """
        params: dict[str, str] = {
            "run_name": _truncate(run_name, 500),
            "eval_run_id": run_info.run_id,
        }

        if self._system_config is not None:
            cfg = self._system_config
            params["judge_model"] = f"{cfg.llm.provider}/{cfg.llm.model}"
            params["embedding_model"] = (
                f"{cfg.embedding.provider}/{cfg.embedding.model}"
            )

            if cfg.judge_panel is not None:
                params["judge_panel"] = ", ".join(cfg.judge_panel.judges)
                params["judge_aggregation"] = cfg.judge_panel.aggregation_strategy

            if cfg.agents is not None:
                params["agents_enabled"] = str(cfg.agents.enabled)

        self._client.log_params(params)

    def _register_models(self) -> None:
        """Register judge/embedding/agent models for the MLflow Models UI.

        Logs a lightweight pyfunc artifact per unique ``provider>model`` and
        registers a model version on this run. Failures are logged and do not
        abort finalization.
        """
        models = self._collect_models_for_registry()
        if not models:
            return

        model_ref = _build_eval_model_ref()
        seen: set[str] = set()
        for role, provider, model in models:
            registry_name = _registry_model_name(provider, model)
            if not registry_name or registry_name in seen:
                continue
            seen.add(registry_name)

            artifact_path = (
                f"registered_models/"
                f"{_sanitize_artifact_segment(role)}_"
                f"{_sanitize_artifact_segment(provider)}_"
                f"{_sanitize_artifact_segment(model)}"
            )
            try:
                self._client.pyfunc.log_model(
                    artifact_path=artifact_path,
                    python_model=model_ref,
                    registered_model_name=registry_name,
                    pip_requirements=["mlflow"],
                    metadata={
                        "role": role,
                        "provider": provider,
                        "model": model,
                        "logical_name": f"{provider}/{model}",
                    },
                )
                logger.info("mlflow: registered model %s (%s)", registry_name, role)
            except _MLFLOW_ERRORS:
                logger.exception(
                    "mlflow: failed to register model %s (%s)", registry_name, role
                )

    def _collect_models_for_registry(self) -> list[tuple[str, str, str]]:
        """Collect (role, provider, model) triples from system config."""
        cfg = self._system_config
        if cfg is None:
            return []

        models: list[tuple[str, str, str]] = []

        try:
            for judge_id, llm_cfg in cfg.get_judge_configs():
                models.append((f"judge:{judge_id}", llm_cfg.provider, llm_cfg.model))
        except ConfigurationError:
            models.append(("judge", cfg.llm.provider, cfg.llm.model))

        models.append(("embedding", cfg.embedding.provider, cfg.embedding.model))

        if cfg.agents is not None:
            for agent_name, agent_def in cfg.agents.agents.items():
                provider = getattr(agent_def, "provider", None)
                model = getattr(agent_def, "model", None)
                if provider and model:
                    models.append((f"agent:{agent_name}", str(provider), str(model)))

        return models

    def _log_result_metrics(self, result: EvaluationResult) -> None:
        """Log all metrics for a single result at the current step.

        Builds the metrics dict first, logs to MLflow, then commits
        accumulator changes and the table row only on success.
        """
        acc = self._acc
        step = acc.step
        metric_id = _sanitize_metric_key(result.metric_identifier)
        metrics: dict[str, float] = {}

        score_val: Optional[float] = None
        if result.score is not None:
            score_val = float(result.score)
            metrics[_metric_key("score/", metric_id)] = score_val

        pass_val = 1.0 if result.result == "PASS" else 0.0
        metrics[_metric_key("pass/", metric_id)] = pass_val

        if result.evaluation_latency > 0:
            metrics[_metric_key("latency/", metric_id)] = result.evaluation_latency

        if result.execution_time > 0:
            metrics[_metric_key("execution_time/", metric_id)] = result.execution_time

        if result.agent_latency > 0:
            metrics[_metric_key("agent_latency/", metric_id)] = result.agent_latency

        pending_tokens = self._collect_token_metrics(result, metrics, metric_id)
        self._collect_judge_and_streaming_metrics(result, metrics, metric_id)

        self._client.log_metrics(metrics, step=step)

        acc.step += 1
        pm = acc.per_metric.setdefault(metric_id, _PerMetricAccumulator())
        if score_val is not None:
            acc.scores.append(score_val)
            pm.scores.append(score_val)
        acc.pass_values.append(pass_val)
        pm.pass_values.append(pass_val)
        if result.evaluation_latency > 0:
            acc.latencies.append(result.evaluation_latency)
        for tok_key, tok_val in pending_tokens.items():
            acc.token_totals[tok_key] = acc.token_totals.get(tok_key, 0.0) + tok_val

        self._accumulate_table_row(result)

    def _log_result_trace(self, result: EvaluationResult) -> None:
        """Log an evaluation result as an MLflow trace for row-level analysis.

        Each result becomes a trace visible in the MLflow Traces tab with
        all evaluation fields.
        """
        trace_name = f"{result.metric_identifier} | {result.conversation_group_id}"
        if result.turn_id:
            trace_name += f" | {result.turn_id}"

        with self._client.start_span(name=trace_name) as span:
            span.set_inputs(
                {
                    "conversation_group_id": result.conversation_group_id,
                    "turn_id": result.turn_id or "",
                    "metric_identifier": result.metric_identifier,
                    "query": result.query or "",
                    "response": result.response or "",
                    "contexts": result.contexts or "",
                    "tool_calls": result.tool_calls or "",
                    "expected_response": _format_expected_response(
                        result.expected_response
                    ),
                    "expected_intent": result.expected_intent or "",
                    "expected_keywords": result.expected_keywords or "",
                    "expected_tool_calls": result.expected_tool_calls or "",
                }
            )
            span.set_outputs(
                {
                    "result": result.result,
                    "score": result.score,
                    "threshold": (
                        result.threshold if result.threshold is not None else 0.0
                    ),
                    "reason": _truncate(result.reason, 4000) if result.reason else "",
                    "judge_scores": _format_judge_scores(result.judge_scores),
                }
            )
            span.set_attributes(
                {
                    "conversation_group_id": result.conversation_group_id,
                    "turn_id": result.turn_id or "",
                    "metric_identifier": result.metric_identifier,
                    "metric_metadata": result.metric_metadata or "",
                    "threshold": (
                        result.threshold if result.threshold is not None else 0.0
                    ),
                    "evaluation_latency": result.evaluation_latency,
                    "execution_time": result.execution_time,
                    "api_input_tokens": result.api_input_tokens,
                    "api_output_tokens": result.api_output_tokens,
                    "judge_llm_input_tokens": result.judge_llm_input_tokens,
                    "judge_llm_output_tokens": result.judge_llm_output_tokens,
                }
            )

    def _accumulate_table_row(self, result: EvaluationResult) -> None:
        """Accumulate a result dict for the results table artifact."""
        max_col_len = 1000
        self._acc.table_rows.append(
            {
                "conversation_group_id": result.conversation_group_id,
                "turn_id": result.turn_id or "",
                "metric": result.metric_identifier,
                "result": result.result,
                "score": result.score if result.score is not None else None,
                "threshold": result.threshold if result.threshold is not None else None,
                "reason": (
                    _truncate(result.reason, max_col_len) if result.reason else ""
                ),
                "query": (_truncate(result.query, max_col_len) if result.query else ""),
                "response": (
                    _truncate(result.response, max_col_len) if result.response else ""
                ),
                "expected_response": _truncate(
                    _format_expected_response(result.expected_response), max_col_len
                ),
                "execution_time": result.execution_time,
                "evaluation_latency": result.evaluation_latency,
                "agent_latency": result.agent_latency,
                "time_to_first_token": result.time_to_first_token,
                "streaming_duration": result.streaming_duration,
                "tokens_per_second": result.tokens_per_second,
                "api_input_tokens": result.api_input_tokens,
                "api_output_tokens": result.api_output_tokens,
                "judge_llm_input_tokens": result.judge_llm_input_tokens,
                "judge_llm_output_tokens": result.judge_llm_output_tokens,
                "embedding_tokens": result.embedding_tokens,
                "judge_scores": _format_judge_scores(result.judge_scores),
            }
        )

    @staticmethod
    def _collect_token_metrics(
        result: EvaluationResult, metrics: dict[str, float], metric_id: str
    ) -> dict[str, float]:
        """Build token usage metrics and return pending totals to commit later."""
        token_fields = (
            ("tokens/api_input", result.api_input_tokens),
            ("tokens/api_output", result.api_output_tokens),
            ("tokens/judge_input", result.judge_llm_input_tokens),
            ("tokens/judge_output", result.judge_llm_output_tokens),
            ("tokens/embedding", result.embedding_tokens),
        )
        pending_totals: dict[str, float] = {}
        for key, value in token_fields:
            if value > 0:
                metrics[_metric_key(f"{key}/", metric_id)] = float(value)
                pending_totals[key] = float(value)
        return pending_totals

    def _collect_judge_and_streaming_metrics(
        self, result: EvaluationResult, metrics: dict[str, float], metric_id: str
    ) -> None:
        """Collect per-judge scores and streaming performance metrics."""
        if result.judge_scores:
            for judge_score in result.judge_scores:
                if judge_score.score is not None:
                    judge_key = _sanitize_metric_key(judge_score.judge_id)
                    metrics[_metric_key(f"judge/{judge_key}/", metric_id)] = float(
                        judge_score.score
                    )

        if result.time_to_first_token and result.time_to_first_token > 0:
            metrics[_metric_key("time_to_first_token/", metric_id)] = (
                result.time_to_first_token
            )
        if result.streaming_duration and result.streaming_duration > 0:
            metrics[_metric_key("streaming_duration/", metric_id)] = (
                result.streaming_duration
            )
        if result.tokens_per_second and result.tokens_per_second > 0:
            metrics[_metric_key("tokens_per_second/", metric_id)] = (
                result.tokens_per_second
            )

    def _log_aggregate_metrics(self) -> None:
        """Compute and log aggregate metrics at the end of the run.

        Logs both overall aggregates (mean_score, pass_rate) and per-metric
        aggregates (pass_rate/<metric>, mean_score/<metric>) for cross-run
        comparison in the MLflow UI.
        """
        acc = self._acc
        if acc.step == 0:
            logger.info("mlflow: no results to aggregate; skipping")
            return

        aggregate: dict[str, float] = {
            "aggregate/result_count": float(acc.step),
        }

        if acc.scores:
            aggregate["aggregate/mean_score"] = sum(acc.scores) / len(acc.scores)

        if acc.pass_values:
            aggregate["aggregate/pass_rate"] = sum(acc.pass_values) / len(
                acc.pass_values
            )

        if acc.latencies:
            aggregate["aggregate/mean_latency"] = sum(acc.latencies) / len(
                acc.latencies
            )

        for metric_id, pm in acc.per_metric.items():
            if pm.scores:
                aggregate[_metric_key("aggregate/mean_score/", metric_id)] = sum(
                    pm.scores
                ) / len(pm.scores)
            if pm.pass_values:
                aggregate[_metric_key("aggregate/pass_rate/", metric_id)] = sum(
                    pm.pass_values
                ) / len(pm.pass_values)

        token_aggregate_keys = {
            "tokens/api_input": "aggregate/total_api_input_tokens",
            "tokens/api_output": "aggregate/total_api_output_tokens",
            "tokens/judge_input": "aggregate/total_judge_input_tokens",
            "tokens/judge_output": "aggregate/total_judge_output_tokens",
            "tokens/embedding": "aggregate/total_embedding_tokens",
        }
        for token_key, agg_key in token_aggregate_keys.items():
            total = acc.token_totals.get(token_key, 0.0)
            if total > 0:
                aggregate[agg_key] = total

        self._client.log_metrics(aggregate)
        logger.info(
            "MLflow backend finalized: %d results logged (run_id=%s)",
            acc.step,
            self._run_info.run_id if self._run_info else "unknown",
        )

    def _log_results_table(self) -> None:
        """Log accumulated results as a table artifact for columnar viewing.

        Converts the list of row dicts into a columnar dict (column name to
        list of values) which is the format ``mlflow.log_table()`` expects.
        Creates a browsable table in the MLflow Artifacts tab with one row
        per evaluation result and separate columns for each field.
        """
        rows = self._acc.table_rows
        if not rows:
            return

        columnar: dict[str, list[Any]] = {
            key: [row.get(key) for row in rows] for key in rows[0]
        }

        try:
            self._client.log_table(
                data=columnar,
                artifact_file="evaluation_results.json",
            )
            logger.info("mlflow: logged results table (%d rows)", len(rows))
        except _MLFLOW_ERRORS:
            logger.exception("mlflow: failed to log results table")


_MAX_METRIC_KEY_LEN = 250


def _build_eval_model_ref() -> Any:
    """Build a minimal pyfunc model used to link registry entries to a run.

    Built with ``type()`` so the base class can come from a dynamic
    ``importlib`` load (mypy rejects variable base classes in ``class`` stmts).
    """
    # Typed as Any: importlib returns ModuleType, which has no PythonModel attr for mypy.
    pyfunc_mod: Any = importlib.import_module("mlflow.pyfunc")
    python_model_cls = pyfunc_mod.PythonModel

    def predict(
        self: Any,
        context: Any,
        model_input: Any,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Return input unchanged — used only for registry linkage."""
        _ = self, context, params
        return model_input

    def describe(self: Any) -> str:
        """Return a short description of this registry stub."""
        _ = self
        return "lightspeed-evaluation registry stub"

    eval_model_ref_cls = type(
        "EvalModelRef",
        (python_model_cls,),
        {"predict": predict, "describe": describe},
    )
    return eval_model_ref_cls()


def _registry_model_name(provider: str, model: str) -> str:
    """Build an MLflow registered-model name as ``provider>model``.

    MLflow rejects ``/`` and ``:`` in registered model names, so ``>`` is used
    as the separator while preserving the provider/model identity.
    """
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        return ""
    return _truncate(f"{provider}>{model}", _MAX_METRIC_KEY_LEN)


def _sanitize_artifact_segment(name: str) -> str:
    """Sanitize a path segment for MLflow artifact paths."""
    sanitized = name.replace("/", "_").replace(":", "_").replace(">", "_")
    sanitized = sanitized.replace(" ", "_")
    return _truncate(sanitized, 80)


def _sanitize_metric_key(name: str) -> str:
    """Sanitize a metric name for use as an MLflow metric key.

    MLflow metric keys allow alphanumerics, underscores, dashes, periods,
    spaces, and slashes. Replace colons with underscores for compatibility.
    """
    sanitized = name.replace(":", "_")
    return _truncate(sanitized, _MAX_METRIC_KEY_LEN)


def _metric_key(prefix: str, metric_id: str) -> str:
    """Compose and sanitize a full MLflow metric key from prefix and metric id.

    Ensures the final composed key (e.g. ``aggregate/mean_score/ragas_faithfulness``)
    does not exceed MLflow's 250-character limit.
    """
    return _truncate(f"{prefix}{metric_id}", _MAX_METRIC_KEY_LEN)


def _format_expected_response(value: Optional[str | list[str]]) -> str:
    """Format expected_response which can be a string or list of strings."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n---\n".join(str(x) for x in value)
    return str(value)


def _format_judge_scores(judge_scores: Any) -> str:
    """Format judge scores as a readable string for trace output."""
    if not judge_scores:
        return ""
    parts: list[str] = []
    for js in judge_scores:
        score_str = f"{js.judge_id}: {js.score}"
        if js.reason:
            score_str += f" ({_truncate(js.reason, 200)})"
        parts.append(score_str)
    return " | ".join(parts)


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string with ellipsis if it exceeds max_len."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
