"""Langfuse payload conversion into internal trace models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from lightspeed_evaluation.core.models.trace import (
    LLMSpanData,
    Span,
    SpanStatus,
    SpanType,
    Trace,
    TracePlatform,
    normalize_span_type,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.core.tracing._helpers import (
    _as_datetime,
    _coerce_float,
    _coerce_non_negative_float,
    _coerce_non_negative_int,
    _correlated_trace_id,
    _first_present,
    _latency_seconds,
    _metadata_extras,
    _payload_spans,
    _require_ids,
    _SourceFields,
    _tags,
    _text,
)
from lightspeed_evaluation.core.tracing._span_mapping import (
    _llm_span_data,
    _model_info,
    _semantic_data,
    _standard_attributes,
    _token_usage,
)


def _status_from_langfuse(level: Any, status_message: Any = None) -> SpanStatus:
    normalized = str(level or "").strip().upper()
    if normalized == "ERROR":
        return SpanStatus.ERROR
    if status_message or normalized in {"DEFAULT", "DEBUG", "WARNING"}:
        return SpanStatus.OK
    return SpanStatus.UNSET


def _langfuse_latency(
    raw: dict[str, Any],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> Optional[float]:
    """Langfuse latency is already reported in seconds."""
    return _latency_seconds(start_time, end_time, explicit_seconds=raw.get("latency"))


def _langfuse_generation(fields: _SourceFields) -> Optional[LLMSpanData]:
    """Normalize nested Langfuse generation fields without losing extras."""
    usage_key, usage = fields.nested("usage", "usageDetails")
    parameters_key, parameters = fields.nested("modelParameters")
    model = fields.take("model", convert=_text)
    if model is None:
        model = parameters.take("model", convert=_text)
    llm = _llm_span_data(
        model=_model_info(
            model, temperature=parameters.take("temperature", convert=_coerce_float)
        ),
        token_usage=_token_usage(
            usage.take("input", "promptTokens", convert=_coerce_non_negative_int),
            usage.take("output", "completionTokens", convert=_coerce_non_negative_int),
            usage.take("total", "totalTokens", convert=_coerce_non_negative_int),
        ),
        cost=fields.take(
            "calculatedTotalCost",
            "totalCost",
            "cost",
            convert=_coerce_non_negative_float,
        ),
    )
    fields.finish_nested(usage_key, usage)
    fields.finish_nested(parameters_key, parameters)
    return llm


def _langfuse_metadata(
    metadata: _SourceFields, fields: _SourceFields
) -> dict[str, Any]:
    """Preserve custom metadata even when its keys collide with source fields."""
    extras = metadata.remaining("langfuse")
    remaining = fields.remaining("langfuse")
    if extras.keys() & remaining.keys():
        # Source fields have fixed names; retain custom values in their container.
        extras["langfuse.metadata"] = dict(metadata.values)
    return extras | remaining


def span_from_langfuse(raw: dict[str, Any], *, trace_id: str | None = None) -> Span:
    """Convert a Langfuse observation with source-specific field extraction."""
    resolved_trace_id, span_id = _require_ids(
        trace_id=_correlated_trace_id(
            trace_id, raw.get("traceId"), raw.get("trace_id")
        ),
        span_id=raw.get("id") or raw.get("observationId") or raw.get("span_id"),
        missing_trace="Langfuse observation is missing traceId",
        missing_span="Langfuse observation is missing id",
    )
    fields = _SourceFields(
        {
            key: raw[key]
            for key in (
                "input",
                "inputs",
                "output",
                "outputs",
                "model",
                "calculatedTotalCost",
                "totalCost",
                "cost",
                "usage",
                "usageDetails",
                "modelParameters",
            )
            if key in raw
        }
    )
    metadata = _SourceFields(_metadata_extras(raw.get("metadata"), "langfuse"))
    span_type = normalize_span_type(raw.get("type") or raw.get("observationType"))
    start_time = _as_datetime(
        _first_present(raw.get("startTime"), raw.get("start_time"))
    )
    end_time = _as_datetime(_first_present(raw.get("endTime"), raw.get("end_time")))
    parent = _first_present(
        raw.get("parentObservationId"), raw.get("parent_observation_id")
    )
    name = str(raw.get("name") or span_id)
    status_message = _first_present(raw.get("statusMessage"), raw.get("status_message"))
    status = _status_from_langfuse(raw.get("level"), status_message)
    llm = _langfuse_generation(fields)
    return Span(
        span_id=span_id,
        trace_id=resolved_trace_id,
        parent_span_id=str(parent) if parent is not None else None,
        source=TracePlatform.LANGFUSE,
        name=name,
        span_type=span_type,
        status=status,
        status_message=status_message,
        start_time=start_time,
        end_time=end_time,
        latency=_langfuse_latency(raw, start_time, end_time),
        attributes=_standard_attributes(
            span_type=span_type,
            name=(
                raw.get("name") or metadata.take("tool_name", convert=_text) or name
                if span_type == SpanType.TOOL
                else name
            ),
            data=_semantic_data(
                fields, span_type, ("input", "inputs"), ("output", "outputs")
            ),
            llm=llm,
            status=(status, status_message),
        ),
        metadata=_langfuse_metadata(metadata, fields),
        tags=_tags(raw.get("tags")) or [],
        llm=llm,
    )


def _langfuse_observations(raw: dict[str, Any]) -> list[Any]:
    """Collect observation list from common Langfuse payload shapes."""
    observations = raw.get("observations") or raw.get("observationsList")
    if observations is None and isinstance(raw.get("data"), dict):
        observations = raw["data"].get("observations")
    return observations or []


def trace_from_langfuse(raw: dict[str, Any]) -> Trace:
    """Convert a Langfuse trace (+ observations) payload to :class:`Trace`."""
    trace_id = raw.get("id") or raw.get("traceId") or raw.get("trace_id")
    if not trace_id:
        raise DataValidationError("Langfuse trace payload is missing id")
    spans = [
        span_from_langfuse(item, trace_id=str(trace_id))
        for item in _payload_spans(_langfuse_observations(raw))
    ]
    return Trace(
        trace_id=str(trace_id),
        name=raw.get("name"),
        session_id=raw.get("sessionId") or raw.get("session_id"),
        user_id=raw.get("userId") or raw.get("user_id"),
        platform=TracePlatform.LANGFUSE,
        start_time=_as_datetime(
            _first_present(
                raw.get("timestamp"), raw.get("startTime"), raw.get("start_time")
            )
        ),
        end_time=_as_datetime(_first_present(raw.get("endTime"), raw.get("end_time"))),
        metadata=_metadata_extras(raw.get("metadata"), "langfuse"),
        tags=_tags(raw.get("tags")) or [],
        spans=spans,
    )
