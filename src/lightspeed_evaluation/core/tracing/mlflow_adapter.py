"""MLflow payload conversion into internal trace models."""

from __future__ import annotations

from typing import Any, Optional

from lightspeed_evaluation.core.models.trace import (
    Span,
    SpanStatus,
    SpanType,
    Trace,
    TracePlatform,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.core.tracing._helpers import (
    _as_datetime,
    _correlated_trace_id,
    _first_present,
    _metadata_extras,
    _parse_jsonish,
    _payload_spans,
    _require_ids,
    _SourceFields,
    _tags,
)
from lightspeed_evaluation.core.tracing._span_mapping import (
    _known_span_type,
    _semantic_data,
    _source_latency,
    _source_llm,
    _source_tags,
    _span_events,
    _standard_attributes,
)


def _mlflow_identity(
    raw: dict[str, Any],
    context: dict[str, Any],
    fields: _SourceFields,
    trace_id: Optional[str],
    metadata: dict[str, Any],
) -> Any:
    """Resolve MLflow trace identity, preserving a divergent OTel context ID.

    The MLflow request ID is the normalized trace identity. The OTel context
    trace ID is a distinct identifier: a fallback when no request ID exists,
    preserved in metadata when both are present and they differ.
    """
    request_identity = _correlated_trace_id(
        trace_id, fields.take("mlflow.traceRequestId"), raw.get("trace_id")
    )
    context_trace_id = context.get("trace_id")
    if (
        request_identity is not None
        and context_trace_id is not None
        and str(context_trace_id) != request_identity
    ):
        metadata["mlflow.span_context_trace_id"] = context_trace_id
    return request_identity if request_identity is not None else context_trace_id


def _status_from_mlflow(code: Any) -> SpanStatus:
    normalized = str(code or "").strip().upper()
    if normalized in {"OK", "STATUS_CODE_OK"}:
        return SpanStatus.OK
    if normalized in {"ERROR", "STATUS_CODE_ERROR"}:
        return SpanStatus.ERROR
    return SpanStatus.UNSET


def _mlflow_metadata(
    identity_metadata: dict[str, Any],
    fields: _SourceFields,
    raw: dict[str, Any],
    span_type: SpanType,
) -> dict[str, Any]:
    """Namespace unconsumed fields; keep retrieval values as received."""
    metadata = identity_metadata | fields.remaining("mlflow")
    if span_type == SpanType.RETRIEVER:
        _restore_raw_retrieval(metadata, raw, raw.get("attributes") or {})
    return metadata


def _restore_raw_retrieval(
    metadata: dict[str, Any], raw: dict[str, Any], attributes: dict[str, Any]
) -> None:
    """Keep retriever inputs/outputs in metadata as received from the source.

    Attribute values are parsed for semantic mapping, but the retrieval
    contract preserves the original representation (e.g. JSON strings).
    """
    for key in ("mlflow.spanInputs", "inputs", "mlflow.spanOutputs", "outputs"):
        meta_key = key if key.startswith("mlflow.") else f"mlflow.{key}"
        if meta_key not in metadata:
            continue
        if key in attributes:
            metadata[meta_key] = attributes[key]
        elif key in raw:
            metadata[meta_key] = raw[key]


def span_from_mlflow(raw: dict[str, Any], *, trace_id: str | None = None) -> Span:
    """Convert a single MLflow span without discarding unmapped attributes."""
    context = raw.get("context") or {}
    fields = _SourceFields(
        {
            **{key: raw[key] for key in ("inputs", "outputs") if key in raw},
            **{
                str(k): _parse_jsonish(v)
                for k, v in (raw.get("attributes") or {}).items()
            },
        }
    )
    identity_metadata: dict[str, Any] = {}
    status = _status_from_mlflow(raw.get("status_code"))
    status_message = raw.get("status_message")
    resolved_trace_id, span_id = _require_ids(
        trace_id=_mlflow_identity(raw, context, fields, trace_id, identity_metadata),
        span_id=context.get("span_id") or raw.get("span_id"),
        missing_trace="MLflow span is missing trace_id",
        missing_span="MLflow span is missing span_id",
    )
    start_time = _as_datetime(raw.get("start_time"))
    end_time = _as_datetime(raw.get("end_time"))
    span_type = (
        fields.take("mlflow.spanType", "span_type", convert=_known_span_type)
        or SpanType.UNKNOWN
    )
    llm = _source_llm(
        fields,
        {
            "model": ("mlflow.spanModel", "model"),
            "input": ("llm.usage.prompt_tokens", "input_tokens"),
            "output": ("llm.usage.completion_tokens", "output_tokens"),
            "total": ("llm.usage.total_tokens", "total_tokens"),
            "cost": ("cost_usd", "mlflow.spanCost"),
        },
    )
    name = str(raw.get("name") or span_id)
    tags = _source_tags(raw, fields)
    return Span(
        span_id=span_id,
        trace_id=resolved_trace_id,
        parent_span_id=(
            str(raw["parent_id"]) if raw.get("parent_id") is not None else None
        ),
        source=TracePlatform.MLFLOW,
        name=name,
        span_type=span_type,
        status=status,
        status_message=status_message,
        start_time=start_time,
        end_time=end_time,
        latency=_source_latency(fields, start_time, end_time),
        attributes=_standard_attributes(
            span_type=span_type,
            name=name,
            data=_semantic_data(
                fields,
                span_type,
                ("mlflow.spanInputs", "inputs"),
                ("mlflow.spanOutputs", "outputs"),
            ),
            llm=llm,
            status=(status, status_message),
        ),
        metadata=_mlflow_metadata(identity_metadata, fields, raw, span_type),
        events=_span_events(raw.get("events")),
        tags=tags,
        llm=llm,
    )


def trace_from_mlflow(raw: dict[str, Any]) -> Trace:
    """Convert an MLflow trace payload (``{"spans": [...], ...}``) to :class:`Trace`."""
    spans_raw = _payload_spans(
        _first_present(raw.get("spans"), (raw.get("data") or {}).get("spans"), [])
    )

    info = raw.get("info") or {}
    span_request_ids = [
        parsed
        for item in spans_raw
        if (
            parsed := _parse_jsonish(
                (item.get("attributes") or {}).get("mlflow.traceRequestId")
            )
        )
        is not None
    ]
    request_id = _first_present(
        raw.get("trace_id"),
        info.get("request_id"),
        info.get("trace_id"),
        *span_request_ids,
        *[item.get("trace_id") for item in spans_raw],
        *[(item.get("context") or {}).get("trace_id") for item in spans_raw],
    )
    if request_id is None:
        raise DataValidationError("MLflow trace payload is missing trace_id")
    request_id = str(request_id)
    if _first_present(
        raw.get("trace_id"),
        info.get("request_id"),
        info.get("trace_id"),
        *span_request_ids,
    ) is None and any(
        (item.get("context") or {}).get("trace_id") not in (None, request_id)
        for item in spans_raw
    ):
        # Identity derives from OTel context IDs; all spans must agree on them.
        raise DataValidationError("Span trace_id conflicts with enclosing trace")

    tags = info.get("tags") or raw.get("tags") or {}
    tag_list = (
        [f"{k}:{v}" for k, v in tags.items()]
        if isinstance(tags, dict)
        else _tags(tags) or []
    )
    return Trace(
        trace_id=request_id,
        name=info.get("name") or raw.get("name"),
        session_id=_first_present(
            tags.get("mlflow.trace.session") if isinstance(tags, dict) else None,
            raw.get("session_id"),
        ),
        platform=TracePlatform.MLFLOW,
        start_time=_as_datetime(
            _first_present(
                info.get("timestamp_ms"), raw.get("start_time"), info.get("timestamp")
            )
        ),
        end_time=_as_datetime(raw.get("end_time")),
        metadata={
            k: v
            for k, v in {
                "request": _parse_jsonish(raw.get("request")),
                "response": _parse_jsonish(raw.get("response")),
                **_metadata_extras(info.get("metadata"), "mlflow"),
            }.items()
            if v is not None
        },
        tags=tag_list,
        spans=[span_from_mlflow(item, trace_id=request_id) for item in spans_raw],
    )
