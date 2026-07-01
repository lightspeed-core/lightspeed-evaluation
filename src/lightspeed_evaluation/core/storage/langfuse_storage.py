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
    2. ``save_run(results)``    — accumulates all results (called per conversation).
    3. ``finalize()``           — creates a trace span, writes scores, and flushes.
    4. ``close()``              — shuts down the client.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any, Optional

from lightspeed_evaluation.core.models.data import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.storage.config import LangfuseBackendConfig
from lightspeed_evaluation.core.storage.protocol import RunInfo

logger = logging.getLogger(__name__)

_HAS_LANGFUSE = importlib.util.find_spec("langfuse") is not None


class LangfuseStorageBackend:
    """Storage backend that exports evaluation results to Langfuse.

    Creates one Langfuse trace (observation span) per evaluation run and
    one score per evaluation result via ``create_score()``.
    Results with ``score=None`` (ERROR/SKIPPED) are skipped from numeric
    scoring but their status is logged.

    Uses the Langfuse Python SDK v4 API:
    ``start_as_current_observation()``, ``create_score()``, ``flush()``.

    All Langfuse SDK errors are caught and logged — they never fail
    the evaluation pipeline.
    """

    def __init__(self, config: LangfuseBackendConfig) -> None:
        """Initialize the Langfuse storage backend.

        Args:
            config: Langfuse backend configuration with optional host,
                public_key, and secret_key fields.
        """
        self._config = config
        self._client: Any = None
        self._run_info: Optional[RunInfo] = None
        self._results: list[EvaluationResult] = []

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        return "langfuse"

    def initialize(self, run_info: RunInfo) -> None:
        """Create the Langfuse client for this run."""
        self._run_info = run_info
        self._results = []

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
        except (RuntimeError, ValueError, OSError, ConnectionError):
            logger.exception("langfuse: failed to initialize client")
            self._client = None

    def save_result(self, result: EvaluationResult) -> None:
        """Accumulate a single result for batch export at finalize."""
        self._results.append(result)

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Accumulate conversation results for batch export at finalize."""
        self._results.extend(results)

    def set_evaluation_context(
        self, evaluation_data: Optional[list[EvaluationData]] = None
    ) -> None:
        """No-op — Langfuse export does not need the full evaluation dataset."""
        _ = evaluation_data

    def finalize(self) -> None:
        """Create a trace span, write all scores, and flush to Langfuse."""
        if self._client is None:
            return

        if not self._results:
            logger.info("langfuse: no results to report; skipping")
            return

        try:
            self._write_trace_and_scores()
        except (RuntimeError, ValueError, OSError, ConnectionError):
            logger.exception("langfuse: failed to write trace and scores")

    def close(self) -> None:
        """Shut down the Langfuse client."""
        if self._client is not None:
            try:
                self._client.shutdown()
            except (RuntimeError, OSError, ConnectionError):
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

    def _write_trace_and_scores(self) -> None:
        """Create one trace span and emit one score per result row.

        Uses the v4 observation-centric API:
        - ``start_as_current_observation()`` to create the trace span
        - ``create_score()`` to attach scores by trace_id
        - ``flush()`` to ensure all events are sent
        """
        run_name = self._run_info.name if self._run_info else "evaluation"
        trace_name = _truncate(f"lightspeed_eval__{run_name}", 256)

        trace_meta: dict[str, Any] = {
            "run_name": run_name,
            "result_count": len(self._results),
            "rows_preview": self._build_rows_preview(),
        }

        with self._client.start_as_current_observation(
            name=trace_name,
            as_type="span",
            metadata=trace_meta,
        ) as span:
            trace_id = span.trace_id

            for r in self._results:
                if r.score is None:
                    logger.debug(
                        "langfuse: skipping score for %s "
                        "(status=%s, no numeric score)",
                        r.metric_identifier,
                        r.result,
                    )
                    continue

                self._client.create_score(
                    trace_id=trace_id,
                    name=_truncate(r.metric_identifier, 200),
                    value=float(r.score),
                    data_type="NUMERIC",
                    comment=_format_comment(r),
                    metadata=_build_score_metadata(r),
                )

        self._client.flush()

    def _build_rows_preview(self) -> list[dict[str, Any]]:
        """Build a compact preview of the first 50 rows for trace metadata."""
        preview: list[dict[str, Any]] = []
        for i, r in enumerate(self._results[:50]):
            preview.append(
                {
                    "idx": i,
                    "conversation_group_id": r.conversation_group_id,
                    "turn_id": r.turn_id or "",
                    "metric": r.metric_identifier,
                    "result": r.result,
                    "score": r.score,
                }
            )
        return preview


def _format_comment(r: EvaluationResult) -> str:
    """Build a human-readable comment for a Langfuse score entry."""
    parts: list[str] = [
        f"result={r.result}",
        f"conversation_group_id={r.conversation_group_id}",
        f"turn_id={r.turn_id or ''}",
    ]
    if r.reason:
        max_reason = 1200
        reason = (
            r.reason
            if len(r.reason) <= max_reason
            else r.reason[: max_reason - 3] + "..."
        )
        parts.append(f"reason={reason}")
    return " | ".join(parts)


def _build_score_metadata(r: EvaluationResult) -> dict[str, Any]:
    """Build per-score metadata mirroring evaluation CSV fields."""
    max_text = 8000
    return {
        "query": _truncate(r.query, max_text) if r.query else "",
        "response": _truncate(r.response, max_text) if r.response else "",
        "conversation_group_id": r.conversation_group_id,
        "turn_id": r.turn_id or "",
        "tool_calls": _safe_truncate(r.tool_calls, max_text),
        "contexts": _safe_truncate(r.contexts, max_text),
        "expected_response": _format_expected_response(r.expected_response, max_text),
        "expected_intent": _safe_truncate(r.expected_intent, max_text),
        "expected_tool_calls": _safe_truncate(r.expected_tool_calls, max_text),
        "expected_keywords": _safe_truncate(r.expected_keywords, max_text),
    }


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
