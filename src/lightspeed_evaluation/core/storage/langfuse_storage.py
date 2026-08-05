"""Langfuse storage backend for evaluation results.

Implements :class:`~lightspeed_evaluation.core.storage.protocol.BaseStorageBackend`
so Langfuse plugs into the standard pipeline storage lifecycle without any
changes to the runner, API, or pipeline modules.

Install with: ``pip install 'lightspeed-evaluation[langfuse]'``

Requires **Langfuse Python SDK v4** (``langfuse>=4.0.0,<5.0.0``).

Credentials are resolved from :class:`LangfuseBackendConfig` fields first,
then from ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, and
``LANGFUSE_HOST`` environment variables as fallback (standard Langfuse SDK
behavior).

Lifecycle:
    1. ``initialize(run_info)`` — creates the Langfuse client.
    2. ``save_result`` / ``save_run`` — accumulate results and export per-turn
       traces/scores (and optional Dataset links) incrementally.
    3. ``finalize(success)`` — export any remaining turns, write run-level
       aggregates, and record eval status (complete/failed).
    4. ``close()`` — shuts down the client.
"""

from __future__ import annotations

import importlib.util
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ValidationError

from lightspeed_evaluation.core.constants import SUPPORTED_CSV_COLUMNS
from lightspeed_evaluation.core.models.data import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.storage.config import LangfuseBackendConfig
from lightspeed_evaluation.core.storage.protocol import RunInfo

logger = logging.getLogger(__name__)

_HAS_LANGFUSE = importlib.util.find_spec("langfuse") is not None
_MAX_TEXT = 8000

# Per-metric columns excluded from turn-level csv_turn convenience metadata.
_TURN_LEVEL_CSV_EXCLUDE = frozenset(
    {
        "metric_identifier",
        "metric_metadata",
        "result",
        "score",
        "threshold",
        "reason",
        "judge_scores",
    }
)
_TURN_CSV_KEYS = tuple(
    column for column in SUPPORTED_CSV_COLUMNS if column not in _TURN_LEVEL_CSV_EXCLUDE
)


def _langfuse_errors() -> tuple[type[BaseException], ...]:
    """SDK/network errors that must not abort the evaluation pipeline.

    Broader than local constructor failures alone: export paths also raise
    ``ValidationError`` (response parse), ``ApiError``, ``httpx.HTTPError``,
    and timeouts. Shared by ``initialize``, ``save_*``, ``finalize``, and
    ``close`` so lifecycle handling stays consistent.
    """
    errors: list[type[BaseException]] = [
        RuntimeError,
        ValueError,
        TypeError,
        OSError,
        ConnectionError,
        TimeoutError,
        ValidationError,
    ]
    if not _HAS_LANGFUSE:
        return tuple(errors)
    try:
        api_error_mod = importlib.import_module("langfuse.api.core.api_error")
        errors.append(api_error_mod.ApiError)
    except ImportError:
        pass
    try:
        httpx_mod = importlib.import_module("httpx")
        errors.append(httpx_mod.HTTPError)
    except ImportError:
        pass
    return tuple(errors)


_LANGFUSE_ERRORS = _langfuse_errors()


@dataclass
class _ExportState:
    """Mutable per-run export progress for incremental Langfuse writes."""

    exported_turn_keys: set[tuple[str, str]] = field(default_factory=set)
    # (conversation_group_id, turn_id, metric_identifier) already scored.
    exported_metric_keys: set[tuple[str, str, str]] = field(default_factory=set)
    # Existing Langfuse trace id per turn — used to append late metric scores.
    turn_trace_ids: dict[tuple[str, str], str] = field(default_factory=dict)
    # None = not attempted; "" = ensure failed; otherwise dataset name.
    dataset_ready: Optional[str] = None
    turns_written: int = 0
    items_upserted: int = 0
    run_items_created: int = 0


class LangfuseStorageBackend:
    """Storage backend that exports evaluation results to Langfuse.

    Unified flow (aligned with MLflow's incremental + aggregate model):

      1. **Per-turn traces + scores** (incremental in ``save_*``) — one trace per
         ``(conversation_group_id, turn_id)`` with CSV-shaped metadata and one
         ``create_score()`` per metric row (``score=None`` rows are skipped).
      2. **Optional Dataset run** — when ``dataset_name`` is set, upserts Dataset
         items and creates run items linking each turn to a Datasets → Runs entry.
      3. **Run-level aggregates** (finalize) — a summary span with ``aggregate/*``
         scores plus ``eval_status`` / ``eval/success`` from ``finalize(success)``.

    Uses the Langfuse Python SDK v4 API:
    ``start_as_current_observation()``, ``create_score()``,
    ``create_dataset()`` / ``get_dataset()``, ``create_dataset_item()``,
    ``api.dataset_run_items.create()``, ``flush()``.

    All Langfuse SDK errors are caught and logged — they never fail
    the evaluation pipeline.
    """

    def __init__(self, config: LangfuseBackendConfig) -> None:
        """Initialize the Langfuse storage backend.

        Args:
            config: Langfuse backend configuration with optional host,
                public_key, secret_key, and dataset_name fields.
        """
        self._config = config
        self._client: Any = None
        self._run_info: Optional[RunInfo] = None
        self._results: list[EvaluationResult] = []
        self._export = _ExportState()

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        return "langfuse"

    def initialize(self, run_info: RunInfo) -> None:
        """Create the Langfuse client for this run."""
        self._run_info = run_info
        self._results = []
        self._export = _ExportState()

        if not _HAS_LANGFUSE:
            logger.error(
                "langfuse is not installed. "
                "Add: pip install 'lightspeed-evaluation[langfuse]'"
            )
            return

        langfuse_mod = importlib.import_module("langfuse")

        kwargs = self._build_client_kwargs()
        try:
            self._client = langfuse_mod.Langfuse(**kwargs)
        except _LANGFUSE_ERRORS:
            logger.exception("langfuse: failed to initialize client")
            self._client = None

    def save_result(self, result: EvaluationResult) -> None:
        """Accumulate a result; turn export happens in ``save_run`` / ``finalize``.

        Individual ``save_result`` calls may be partial turns (one metric), so
        they are buffered. The evaluation pipeline writes complete conversation
        batches via ``save_run``, which exports turns incrementally. Late metric
        rows for an already-exported turn are appended to that turn's trace in
        ``finalize`` (or a later ``save_run``).
        """
        self._results.append(result)

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Accumulate conversation results and export their turns immediately."""
        if not results:
            return
        self._results.extend(results)
        if self._client is None:
            return
        try:
            self._export_turn_batch(results, flush=True)
        except _LANGFUSE_ERRORS:
            logger.exception("langfuse: failed to export run batch incrementally")

    def set_evaluation_context(
        self, evaluation_data: Optional[list[EvaluationData]] = None
    ) -> None:
        """No-op — Langfuse export does not need the full evaluation dataset."""
        _ = evaluation_data

    def finalize(self, success: bool = True) -> None:
        """Flush remaining turns, write aggregates, and record eval status.

        Args:
            success: ``True`` for a complete eval (``eval_status=complete``);
                ``False`` for an aborted eval (``eval_status=failed``).
                Incremental turn traces and final aggregates are written in
                both cases.
        """
        if self._client is None:
            return

        if not self._results:
            logger.info("langfuse: no results to report; skipping")
            return

        try:
            # Cover save_result-only paths, unexported turns, and late metric rows
            # for turns that were already exported.
            pending = [
                result
                for result in self._results
                if _metric_export_key(result) not in self._export.exported_metric_keys
            ]
            if pending:
                self._export_turn_batch(pending, flush=False)
        except _LANGFUSE_ERRORS:
            logger.exception("langfuse: failed to export remaining turns")

        try:
            self._write_aggregate_summary(self._dataset_run_name(), success=success)
        except _LANGFUSE_ERRORS:
            logger.exception("langfuse: failed to write aggregate summary")

        # Always flush buffered SDK events, even if aggregate write failed.
        try:
            self._client.flush()
        except _LANGFUSE_ERRORS:
            logger.exception("langfuse: flush after finalize failed")

        logger.info(
            "langfuse: exported %d turn trace(s)%s; wrote aggregate summary "
            "(eval_status=%s)",
            self._export.turns_written,
            (
                f", dataset={self._export.dataset_ready!r} "
                f"items={self._export.items_upserted} "
                f"run_items={self._export.run_items_created}"
                if self._export.dataset_ready
                else ""
            ),
            "complete" if success else "failed",
        )

    def close(self) -> None:
        """Shut down the Langfuse client."""
        if self._client is not None:
            try:
                self._client.shutdown()
            except _LANGFUSE_ERRORS:
                logger.debug("langfuse: shutdown raised; ignoring")
            self._client = None

    def _build_client_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for the Langfuse constructor."""
        kwargs: dict[str, Any] = {}
        if self._config.public_key:
            kwargs["public_key"] = self._config.public_key
        if self._config.secret_key:
            kwargs["secret_key"] = self._config.secret_key
        if self._config.host:
            kwargs["host"] = self._config.host.strip()
        return kwargs

    def _run_label(self) -> str:
        """Human-readable base name for this evaluation run."""
        if self._run_info and self._run_info.name:
            return self._run_info.name
        return "evaluation"

    def _dataset_run_name(self) -> str:
        """Unique Langfuse dataset run name (stable across item upserts)."""
        base = self._run_label()
        run_id = self._run_info.run_id if self._run_info else "unknown"
        return _truncate(f"{base}__{run_id[:8]}", 200)

    def _export_turn_batch(
        self, results: list[EvaluationResult], *, flush: bool = True
    ) -> None:
        """Export not-yet-written turns from *results* (traces, scores, links)."""
        if self._client is None or not results:
            return

        export_run_name = self._dataset_run_name()
        dataset_name = self._link_dataset_if_configured()
        run_description = (
            f"lightspeed-evaluation run '{self._run_label()}' "
            f"({len(self._results)} metric row(s))"
        )

        for (conversation_group_id, turn_id), turn_results in _group_results_by_turn(
            results
        ):
            turn_key = (conversation_group_id, turn_id)
            pending_metrics = [
                result
                for result in turn_results
                if _metric_export_key(result) not in self._export.exported_metric_keys
            ]
            if not pending_metrics:
                continue
            try:
                existing_trace_id = self._export.turn_trace_ids.get(turn_key)
                if existing_trace_id:
                    # Late metric rows for an already-exported turn: append scores
                    # to the existing trace so per-turn and aggregate stay aligned.
                    self._append_scores_to_trace(
                        trace_id=existing_trace_id,
                        results=pending_metrics,
                    )
                    continue

                representative = pending_metrics[0]
                turn_trace_id = self._write_turn_trace_and_scores(
                    run_name=export_run_name,
                    representative=representative,
                    turn_results=pending_metrics,
                )
                if not turn_trace_id:
                    continue
                self._export.turn_trace_ids[turn_key] = turn_trace_id
                self._export.exported_turn_keys.add(turn_key)
                self._export.turns_written += 1
                self._mark_metrics_exported(pending_metrics)

                if not dataset_name:
                    continue

                item_id = _item_id(dataset_name, conversation_group_id, turn_id)
                try:
                    self._upsert_dataset_item(
                        dataset_name=dataset_name,
                        item_id=item_id,
                        representative=representative,
                        source_trace_id=turn_trace_id,
                    )
                    self._export.items_upserted += 1
                    self._create_dataset_run_item(
                        {
                            "run_name": export_run_name,
                            "dataset_item_id": item_id,
                            "run_description": run_description,
                            "metadata": {
                                "source": "lightspeed-evaluation",
                                "run_label": self._run_label(),
                                "conversation_group_id": conversation_group_id,
                                "turn_id": turn_id,
                                "metric_count": len(pending_metrics),
                            },
                            "trace_id": turn_trace_id,
                        }
                    )
                    self._export.run_items_created += 1
                except _LANGFUSE_ERRORS:
                    logger.exception(
                        "langfuse: dataset link failed for turn %s/%s; "
                        "continuing remaining turns",
                        conversation_group_id,
                        turn_id or "conversation",
                    )
            except _LANGFUSE_ERRORS:
                logger.exception(
                    "langfuse: turn export failed for %s/%s; "
                    "continuing remaining turns",
                    conversation_group_id,
                    turn_id or "conversation",
                )

        if not flush:
            return
        try:
            self._client.flush()
        except _LANGFUSE_ERRORS:
            logger.exception("langfuse: flush after turn batch failed")

    def _link_dataset_if_configured(self) -> Optional[str]:
        """Ensure dataset exists when configured; cache success/failure per run."""
        if self._export.dataset_ready is not None:
            return self._export.dataset_ready or None

        dataset_name = self._config.dataset_name
        if not dataset_name:
            self._export.dataset_ready = ""
            return None
        try:
            self._ensure_dataset(dataset_name)
            self._export.dataset_ready = dataset_name
            return dataset_name
        except _LANGFUSE_ERRORS:
            logger.exception(
                "langfuse: failed to ensure dataset %r; "
                "continuing with per-turn traces and aggregates only",
                dataset_name,
            )
            self._export.dataset_ready = ""
            return None

    def _write_aggregate_summary(
        self, export_run_name: str, *, success: bool = True
    ) -> None:
        """Emit a run-level aggregate span with summary scores (MLflow-like)."""
        aggregates = _compute_aggregate_scores(self._results)
        eval_status = "complete" if success else "failed"
        trace_name = _truncate(f"lightspeed_eval_aggregate__{export_run_name}", 256)
        rows_preview = self._build_rows_preview()

        with self._client.start_as_current_observation(
            name=trace_name,
            as_type="span",
            input={
                "run_name": export_run_name,
                "result_count": len(self._results),
                "eval_status": eval_status,
            },
            output={"aggregates": aggregates, "rows_preview": rows_preview},
            metadata={
                "run_name": export_run_name,
                "result_count": len(self._results),
                "source": "lightspeed-evaluation",
                "eval_status": eval_status,
            },
        ) as span:
            trace_id = str(span.trace_id)
            for name, value in aggregates.items():
                self._client.create_score(
                    trace_id=trace_id,
                    name=_truncate(name, 200),
                    value=float(value),
                    data_type="NUMERIC",
                    comment=f"aggregate | run={export_run_name}",
                    metadata={
                        "source": "lightspeed-evaluation",
                        "kind": "aggregate",
                        "run_name": export_run_name,
                        "eval_status": eval_status,
                    },
                )
            self._client.create_score(
                trace_id=trace_id,
                name="eval/success",
                value=1.0 if success else 0.0,
                data_type="NUMERIC",
                comment=f"eval_status={eval_status}",
                metadata={
                    "source": "lightspeed-evaluation",
                    "kind": "status",
                    "run_name": export_run_name,
                    "eval_status": eval_status,
                },
            )

    def _upsert_dataset_item(
        self,
        *,
        dataset_name: str,
        item_id: str,
        representative: EvaluationResult,
        source_trace_id: str,
    ) -> None:
        """Upsert a dataset item, tolerating known SDK/server response skew."""
        try:
            self._client.create_dataset_item(
                dataset_name=dataset_name,
                id=item_id,
                input=_build_dataset_item_input(representative),
                expected_output=_build_dataset_item_expected(representative),
                metadata=_build_dataset_item_metadata(representative),
                source_trace_id=source_trace_id,
            )
        except ValidationError as exc:
            # Langfuse Python SDK v4 requires media_references on DatasetItem;
            # some self-hosted OSS servers omit it even after a successful create.
            if _is_missing_field_validation_error(exc, "media_references"):
                logger.warning(
                    "langfuse: dataset item %r upsert likely succeeded; "
                    "ignoring SDK parse error for missing media_references "
                    "(self-hosted Langfuse/SDK version skew)",
                    item_id,
                )
                return
            raise

    def _create_dataset_run_item(self, create_kwargs: dict[str, Any]) -> None:
        """Create a dataset run item, tolerating known SDK/server response skew."""
        dataset_item_id = create_kwargs.get("dataset_item_id", "")
        try:
            self._client.api.dataset_run_items.create(**create_kwargs)
        except ValidationError as exc:
            # Same class of skew as dataset items: HTTP 2xx then strict parse fails.
            if _is_missing_field_validation_error(exc, "media_references"):
                logger.warning(
                    "langfuse: dataset run item for %r likely created; "
                    "ignoring SDK parse error for missing media_references "
                    "(self-hosted Langfuse/SDK version skew)",
                    dataset_item_id,
                )
                return
            raise

    def _write_turn_trace_and_scores(
        self,
        *,
        run_name: str,
        representative: EvaluationResult,
        turn_results: list[EvaluationResult],
    ) -> Optional[str]:
        """Create a per-turn trace with CSV fields and attach metric scores."""
        turn_key = representative.turn_id or "conversation"
        span_name = _truncate(
            f"eval__{representative.conversation_group_id}__{turn_key}",
            256,
        )
        csv_rows = [_result_to_csv_fields(result) for result in turn_results]
        turn_meta = {
            "run_name": run_name,
            "conversation_group_id": representative.conversation_group_id,
            "turn_id": representative.turn_id or "",
            "metric_identifiers": [r.metric_identifier for r in turn_results],
            # Full CSV-equivalent rows for this turn (one object per metric).
            "csv_rows": csv_rows,
            # Convenience: shared turn-level fields from the first metric row.
            "csv_turn": {
                key: csv_rows[0].get(key)
                for key in _TURN_CSV_KEYS
                if key in csv_rows[0]
            },
        }

        with self._client.start_as_current_observation(
            name=span_name,
            as_type="span",
            input=_build_dataset_item_input(representative),
            output={
                "response": (
                    _truncate(representative.response, _MAX_TEXT)
                    if representative.response
                    else ""
                ),
            },
            metadata=turn_meta,
        ) as span:
            trace_id = str(span.trace_id)
            for result in turn_results:
                self._write_score(trace_id=trace_id, result=result)

        return trace_id

    def _append_scores_to_trace(
        self, *, trace_id: str, results: list[EvaluationResult]
    ) -> None:
        """Attach late metric scores to an existing turn trace."""
        for result in results:
            self._write_score(trace_id=trace_id, result=result)
            self._export.exported_metric_keys.add(_metric_export_key(result))

    def _mark_metrics_exported(self, results: list[EvaluationResult]) -> None:
        """Record metric rows as exported so finalize does not skip/re-emit them."""
        for result in results:
            self._export.exported_metric_keys.add(_metric_export_key(result))

    def _write_score(self, *, trace_id: str, result: EvaluationResult) -> None:
        """Emit one numeric score for a result, or skip when score is missing."""
        if result.score is None:
            logger.debug(
                "langfuse: skipping score for %s (status=%s, no numeric score)",
                result.metric_identifier,
                result.result,
            )
            return

        self._client.create_score(
            trace_id=trace_id,
            name=_truncate(result.metric_identifier, 200),
            value=float(result.score),
            data_type="NUMERIC",
            comment=_format_comment(result),
            metadata=_result_to_csv_fields(result),
        )

    def _ensure_dataset(self, name: str) -> None:
        """Get an existing Langfuse Dataset or create it if missing."""
        try:
            self._client.get_dataset(name)
            logger.debug("langfuse: reusing existing dataset %r", name)
            return
        except _LANGFUSE_ERRORS:
            logger.debug(
                "langfuse: dataset %r not found or unavailable; creating",
                name,
                exc_info=True,
            )

        try:
            self._client.create_dataset(
                name=name,
                metadata={"source": "lightspeed-evaluation"},
            )
            logger.info("langfuse: created dataset %r", name)
        except _LANGFUSE_ERRORS:
            # Race or name already exists — verify via get.
            self._client.get_dataset(name)
            logger.debug("langfuse: dataset %r already exists after create race", name)

    def _build_rows_preview(self) -> list[dict[str, Any]]:
        """Build a compact preview of the first 50 rows for trace metadata."""
        preview: list[dict[str, Any]] = []
        for i, result in enumerate(self._results[:50]):
            preview.append(
                {
                    "idx": i,
                    "conversation_group_id": result.conversation_group_id,
                    "turn_id": result.turn_id or "",
                    "metric": result.metric_identifier,
                    "result": result.result,
                    "score": result.score,
                }
            )
        return preview


def _is_missing_field_validation_error(exc: ValidationError, field_name: str) -> bool:
    """Return True when *exc* is a missing required field named *field_name*."""
    for err in exc.errors():
        if err.get("type") != "missing":
            continue
        loc = err.get("loc") or ()
        if field_name in loc:
            return True
    return False


def _metric_export_key(result: EvaluationResult) -> tuple[str, str, str]:
    """Stable key for whether a metric row has been written to Langfuse."""
    return (
        result.conversation_group_id,
        result.turn_id or "",
        result.metric_identifier,
    )


def _compute_aggregate_scores(results: list[EvaluationResult]) -> dict[str, float]:
    """Build MLflow-like aggregate score map from evaluation results."""
    scores = [float(r.score) for r in results if r.score is not None]
    pass_values = [
        1.0 if r.result == "PASS" else 0.0
        for r in results
        if r.result in ("PASS", "FAIL")
    ]
    aggregates: dict[str, float] = {
        "aggregate/result_count": float(len(results)),
    }
    if scores:
        aggregates["aggregate/mean_score"] = sum(scores) / len(scores)
    if pass_values:
        aggregates["aggregate/pass_rate"] = sum(pass_values) / len(pass_values)

    per_metric: OrderedDict[str, list[EvaluationResult]] = OrderedDict()
    for result in results:
        per_metric.setdefault(result.metric_identifier, []).append(result)

    for metric_id, metric_results in per_metric.items():
        metric_scores = [float(r.score) for r in metric_results if r.score is not None]
        metric_passes = [
            1.0 if r.result == "PASS" else 0.0
            for r in metric_results
            if r.result in ("PASS", "FAIL")
        ]
        safe_metric = _truncate(metric_id, 120)
        if metric_scores:
            aggregates[f"aggregate/mean_score/{safe_metric}"] = sum(
                metric_scores
            ) / len(metric_scores)
        if metric_passes:
            aggregates[f"aggregate/pass_rate/{safe_metric}"] = sum(metric_passes) / len(
                metric_passes
            )
    return aggregates


def _group_results_by_turn(
    results: list[EvaluationResult],
) -> list[tuple[tuple[str, str], list[EvaluationResult]]]:
    """Group metric rows by conversation/turn, preserving first-seen order."""
    groups: OrderedDict[tuple[str, str], list[EvaluationResult]] = OrderedDict()
    for result in results:
        key = (result.conversation_group_id, result.turn_id or "")
        groups.setdefault(key, []).append(result)
    return list(groups.items())


def _item_id(dataset_name: str, conversation_group_id: str, turn_id: str) -> str:
    """Build a project-unique Langfuse dataset item id."""
    return _truncate(
        f"{dataset_name}::{conversation_group_id}::{turn_id or 'conversation'}",
        256,
    )


def _result_to_csv_fields(result: EvaluationResult) -> dict[str, Any]:
    """Serialize an evaluation result using the same columns as detailed CSV."""
    row: dict[str, Any] = {}
    for column in SUPPORTED_CSV_COLUMNS:
        if not hasattr(result, column):
            continue
        value = getattr(result, column)
        if column == "judge_scores" and value is not None:
            row[column] = [js.model_dump(mode="json") for js in value]
        elif column == "tag" and value is not None:
            row[column] = sorted(value) if isinstance(value, set) else value
        elif column == "expected_response":
            row[column] = _format_expected_response(value, _MAX_TEXT)
        elif isinstance(value, str):
            row[column] = _truncate(value, _MAX_TEXT)
        else:
            row[column] = value
    return row


def _build_dataset_item_input(result: EvaluationResult) -> dict[str, Any]:
    """Build Langfuse dataset item input from an evaluation result."""
    return {
        "query": _truncate(result.query, _MAX_TEXT) if result.query else "",
        "conversation_group_id": result.conversation_group_id,
        "turn_id": result.turn_id or "",
    }


def _build_dataset_item_expected(result: EvaluationResult) -> dict[str, Any]:
    """Build Langfuse dataset item expected_output from expected_* fields."""
    expected: dict[str, Any] = {}
    expected_response = _format_expected_response(result.expected_response, _MAX_TEXT)
    if expected_response:
        expected["expected_response"] = expected_response
    expected_intent = _safe_truncate(result.expected_intent, _MAX_TEXT)
    if expected_intent:
        expected["expected_intent"] = expected_intent
    expected_tool_calls = _safe_truncate(result.expected_tool_calls, _MAX_TEXT)
    if expected_tool_calls:
        expected["expected_tool_calls"] = expected_tool_calls
    expected_keywords = _safe_truncate(result.expected_keywords, _MAX_TEXT)
    if expected_keywords:
        expected["expected_keywords"] = expected_keywords
    return expected


def _build_dataset_item_metadata(result: EvaluationResult) -> dict[str, Any]:
    """Build stable dataset-item metadata (not run-specific outputs)."""
    return {
        "tags": sorted(result.tag) if result.tag else [],
        "source": "lightspeed-evaluation",
    }


def _format_comment(result: EvaluationResult) -> str:
    """Build a human-readable comment for a Langfuse score entry."""
    parts: list[str] = [
        f"result={result.result}",
        f"conversation_group_id={result.conversation_group_id}",
        f"turn_id={result.turn_id or ''}",
    ]
    if result.reason:
        max_reason = 1200
        reason = (
            result.reason
            if len(result.reason) <= max_reason
            else result.reason[: max_reason - 3] + "..."
        )
        parts.append(f"reason={reason}")
    return " | ".join(parts)


def _safe_truncate(value: Optional[str], max_len: int) -> str:
    """Truncate a nullable string, returning empty string for None."""
    if value is None or not str(value).strip():
        return ""
    return _truncate(str(value), max_len)


def _format_expected_response(value: str | list[str] | None, max_len: int) -> str:
    """Format expected_response which can be a string or list of strings."""
    if value is None:
        return ""
    if isinstance(value, list):
        text = "\n---\n".join(str(x) for x in value)
    else:
        text = str(value)
    return _truncate(text, max_len)


def _truncate(s: str, max_len: int) -> str:
    """Truncate a string with ellipsis if it exceeds max_len."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
