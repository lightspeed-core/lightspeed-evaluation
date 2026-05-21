"""Optional Langfuse export for evaluation results.

Install with: ``pip install 'lightspeed-evaluation[langfuse]'``

Requires Langfuse v2 (``langfuse>=2,<3``) per :class:`langfuse.Langfuse` API
used here (trace and scores).

Environment variables (also read by the Langfuse client if arguments are
omitted): ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``, ``LANGFUSE_HOST``.

Enable export by adding a ``storage`` entry with ``type: langfuse`` and a
required ``host`` in ``system.yaml`` (see :func:`build_langfuse_on_complete_from_storage_configs`).
That ``host`` is always used for the API base for that entry; ``LANGFUSE_HOST`` is not used there.
For programmatic use without YAML, pass ``on_complete=build_langfuse_on_complete_callback(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from lightspeed_evaluation.core.models import EvaluationResult, EvaluationRunContext
from lightspeed_evaluation.core.storage.config import StorageBackendConfig
from lightspeed_evaluation.core.storage.factory import get_langfuse_storage_config

logger = logging.getLogger(__name__)

try:
    from langfuse import Langfuse
except ImportError:  # pragma: no cover - requires optional extra
    Langfuse = None


@dataclass(frozen=True, slots=True)
class LangfuseClientConfig:
    """Optional explicit client kwargs for ``Langfuse(...)``."""

    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: Optional[str] = None


def client_config_from_env() -> LangfuseClientConfig:
    """Build :class:`LangfuseClientConfig` from ``LANGFUSE_*`` environment variables."""
    return LangfuseClientConfig(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST"),
    )


# --- Public API -----------------------------------------------------------------


def push_evaluation_results_to_langfuse(
    results: list[EvaluationResult],
    context: EvaluationRunContext,
    *,
    client_config: Optional[LangfuseClientConfig] = None,
    **deprecated_client_kwargs: Optional[str],
) -> None:
    """Create one Langfuse trace for the run and one score per evaluation row.

    Uses numeric ``value`` from each scored result and rich ``metadata`` per score
    (query/response, ids, tools, expectations, contexts), and a ``comment`` with
    pass/fail status and reason. Row index + ``conversation_group_id`` appear on the
    **trace** ``rows_preview``.
    Rows with ``score=None`` are
    skipped (common for ERROR/SKIPPED). Client initialization failures are
    logged and treated as no-op; write-time SDK errors from
    ``trace/score/flush`` are propagated to the caller.

    Args:
        results: Output of :func:`lightspeed_evaluation.api.evaluate`.
        context: Run context from the ``on_complete`` callback.
        client_config: Optional explicit config for Langfuse constructor.
        **deprecated_client_kwargs: Deprecated keys (``public_key``,
            ``secret_key``, ``host``) kept for compatibility.
    """
    if not results:
        logger.info("langfuse: no results to report; skipping")
        return

    if Langfuse is None:
        logger.error(
            "langfuse is not installed. Add: pip install 'lightspeed-evaluation[langfuse]'"
        )
        return

    runtime_cfg = _resolve_runtime_config(
        client_config=client_config,
        deprecated_client_kwargs=deprecated_client_kwargs,
    )
    client = _create_langfuse_client(runtime_cfg)
    if client is None:
        return
    _write_langfuse_trace_and_scores(client, results, context)


def _resolve_runtime_config(
    *,
    client_config: Optional[LangfuseClientConfig],
    deprecated_client_kwargs: dict[str, Optional[str]],
) -> LangfuseClientConfig:
    """Normalize deprecated convenience kwargs into a single config object."""
    host = deprecated_client_kwargs.get("host")
    public_key = deprecated_client_kwargs.get("public_key")
    secret_key = deprecated_client_kwargs.get("secret_key")
    unsupported = set(deprecated_client_kwargs) - {"host", "public_key", "secret_key"}
    if unsupported:
        logger.warning("langfuse: ignoring unsupported client kwargs: %s", unsupported)

    if client_config is None:
        return LangfuseClientConfig(
            host=host, public_key=public_key, secret_key=secret_key
        )
    return LangfuseClientConfig(
        host=host if host is not None else client_config.host,
        public_key=(public_key if public_key is not None else client_config.public_key),
        secret_key=(secret_key if secret_key is not None else client_config.secret_key),
    )


def _langfuse_init_kwargs_from_config(
    client_config: LangfuseClientConfig,
) -> dict[str, Any]:
    """Build keyword arguments for :class:`langfuse.Langfuse` from our config model."""
    kwargs: dict[str, Any] = {}
    if client_config.public_key is not None:
        kwargs["public_key"] = client_config.public_key
    if client_config.secret_key is not None:
        kwargs["secret_key"] = client_config.secret_key
    if client_config.host is not None:
        stripped = client_config.host.strip()
        if stripped:
            kwargs["host"] = stripped
    return kwargs


def _create_langfuse_client(
    client_config: LangfuseClientConfig,
) -> Optional[Any]:
    """Instantiate the Langfuse SDK client, or log and return ``None`` on failure.

    When ``host`` is set in ``client_config``, it is passed through to ``Langfuse`` so
    the SDK uses that URL as ``base_url`` without mutating ``os.environ`` (thread-safe).
    """
    if Langfuse is None:
        return None

    kwargs = _langfuse_init_kwargs_from_config(client_config)
    try:
        return Langfuse(**kwargs)
    except (RuntimeError, ValueError, OSError, ConnectionError):
        logger.exception("langfuse: failed to initialize client")
        return None


def _write_langfuse_trace_and_scores(
    langfuse: Any,
    results: list[EvaluationResult],
    context: EvaluationRunContext,
) -> None:
    """Create trace, one score per result, flush."""
    rows_preview = _build_trace_rows_preview(results)
    trace_meta: dict[str, Any] = {
        "run_name": context.run_name,
        "result_count": len(results),
        "rows_preview": rows_preview,
    }
    if context.original_data_path is not None:
        trace_meta["original_data_path"] = context.original_data_path
    trace_meta.update(context.metadata or {})

    trace = langfuse.trace(
        name=_truncate(f"lightspeed_eval__{context.run_name}", 256),
        metadata=trace_meta,
    )

    for r in results:
        if r.score is None:
            logger.debug(
                "langfuse: skipping score for %s (status=%s, no numeric score)",
                r.metric_identifier,
                r.result,
            )
            continue
        name = _score_name(r)
        value = float(r.score)
        comment = _format_comment(r)
        trace.score(
            name=name,
            value=value,
            comment=comment,
            metadata=_score_metadata(r),
        )

    langfuse.flush()


def build_langfuse_on_complete_callback(
    *,
    client_config: Optional[LangfuseClientConfig] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
) -> Callable[[list[EvaluationResult], EvaluationRunContext], None]:
    """Build an ``on_complete`` callback for :func:`lightspeed_evaluation.api.evaluate`.

    Args:
        client_config: Optional explicit Langfuse client settings (passed to the
            Langfuse SDK constructor).
        public_key: Deprecated; prefer ``client_config`` or environment variables.
            If set, merged with ``client_config`` when resolving the client.
        secret_key: Deprecated; same merge behavior as ``public_key``.
        host: Deprecated; same merge behavior as ``public_key``.

    Returns:
        A callable invoked after evaluation with ``(results, context)``; it forwards
        to :func:`push_evaluation_results_to_langfuse` and returns ``None``.

    Example::

        from lightspeed_evaluation import evaluate
        from lightspeed_evaluation.integrations.langfuse_reporter import (
            build_langfuse_on_complete_callback,
        )

        on_complete = build_langfuse_on_complete_callback()
        results = evaluate(
            system_config, data, on_complete=on_complete
        )
    """
    return lambda res, ctx: push_evaluation_results_to_langfuse(
        res,
        ctx,
        client_config=client_config,
        public_key=public_key,
        secret_key=secret_key,
        host=host,
    )


def build_langfuse_on_complete_from_storage_configs(
    storage_configs: Sequence[StorageBackendConfig],
) -> Optional[Callable[[list[EvaluationResult], EvaluationRunContext], None]]:
    """Return ``on_complete`` for Langfuse when ``storage`` lists ``type: langfuse``.

    Export is enabled from system storage only (unless you pass
    :func:`build_langfuse_on_complete_callback` explicitly). Add a ``langfuse`` row
    with required ``host`` under ``storage`` in ``system.yaml``. Optional
    ``public_key`` / ``secret_key`` on that row override ``LANGFUSE_PUBLIC_KEY`` /
    ``LANGFUSE_SECRET_KEY``.

    Args:
        storage_configs: Parsed ``SystemConfig.storage`` from the system config.

    Returns:
        Callback for :func:`lightspeed_evaluation.api.evaluate`, or ``None`` when
        no Langfuse storage entry is present.
    """
    entry = get_langfuse_storage_config(storage_configs)
    if entry is None:
        return None
    return build_langfuse_on_complete_callback(
        client_config=LangfuseClientConfig(
            public_key=entry.public_key,
            secret_key=entry.secret_key,
            host=entry.host,
        ),
    )


def _format_comment(r: EvaluationResult) -> str:
    parts: list[str] = [f"result={r.result}"]
    parts.append(f"conversation_group_id={r.conversation_group_id}")
    parts.append(f"turn_id={r.turn_id or ''}")
    if r.reason:
        max_reason = 1200
        reason = (
            r.reason
            if len(r.reason) <= max_reason
            else r.reason[: max_reason - 3] + "..."
        )
        parts.append(f"reason={reason}")
    return " | ".join(parts)


def _score_name(r: EvaluationResult) -> str:
    """Langfuse score name using only metric identifier."""
    return _truncate(r.metric_identifier, 200)


def _score_metadata(r: EvaluationResult) -> dict[str, Any]:
    """Per-score metadata for Langfuse Scores table (mirrors eval CSV-style fields)."""
    max_text = 8000
    return {
        "query": _truncate(r.query, max_text) if r.query else "",
        "response": _truncate(r.response, max_text) if r.response else "",
        "conversation_group_id": r.conversation_group_id,
        "turn_id": r.turn_id or "",
        "tool_calls": _optional_truncate_str(r.tool_calls, max_text),
        "contexts": _optional_truncate_str(r.contexts, max_text),
        "expected_response": _format_expected_response(r.expected_response, max_text),
        "expected_intent": _optional_truncate_str(r.expected_intent, max_text),
        "expected_tool_calls": _optional_truncate_str(r.expected_tool_calls, max_text),
        "expected_keywords": _optional_truncate_str(r.expected_keywords, max_text),
    }


def _optional_truncate_str(value: Optional[str], max_len: int) -> str:
    if value is None or not str(value).strip():
        return ""
    return _truncate(str(value), max_len)


def _format_expected_response(
    value: Optional[Union[str, list[str]]], max_len: int
) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        text = "\n---\n".join(str(x) for x in value)
    else:
        text = str(value)
    return _truncate(text, max_len)


def _build_trace_rows_preview(results: list[EvaluationResult]) -> list[dict[str, Any]]:
    """Trace metadata rows: idx + conversation_group_id key, plus short query/response."""
    preview: list[dict[str, Any]] = []
    for i, r in enumerate(results[:50]):
        preview.append(
            {
                "idx": i,
                "row_key": _evaluation_row_key(i, r.conversation_group_id),
                "conversation_group_id": r.conversation_group_id,
                "turn_id": r.turn_id or "",
                "query": _truncate(r.query, 300),
                "response": _truncate(r.response, 300),
            }
        )
    return preview


def _evaluation_row_key(idx: int, conversation_group_id: str) -> str:
    """Stable id string per evaluation row: ``{idx}_{conversation_group_id}`` (sanitized)."""
    safe = conversation_group_id.replace("/", "_").replace(" ", "_").replace(":", "_")
    return _truncate(f"{idx:04d}_{safe}", 120)


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."
