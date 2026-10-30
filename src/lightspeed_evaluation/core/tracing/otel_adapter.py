"""OpenTelemetry payload conversion into internal trace models."""

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
    _attr_map,
    _coerce_non_negative_int,
    _correlated_trace_id,
    _first_present,
    _metadata_extras,
    _payload_spans,
    _require_ids,
    _SourceFields,
    _tags,
    _text,
    _timestamp,
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


def _status_from_otel(status: Any) -> tuple[SpanStatus, Optional[str]]:
    if not isinstance(status, dict):
        return SpanStatus.UNSET, None
    code = str(status.get("code") or status.get("status_code") or "").upper()
    message = status.get("message") or status.get("description")
    if code in {"OK", "STATUS_CODE_OK", "1"}:
        return SpanStatus.OK, message
    if code in {"ERROR", "STATUS_CODE_ERROR", "2"}:
        return SpanStatus.ERROR, message
    return SpanStatus.UNSET, message


def _otel_type_attributes(fields: _SourceFields, span_type: SpanType) -> dict[str, Any]:
    """Consume optional semantic fields only for applicable span types."""
    mappings = {
        SpanType.GENERATION: (
            "tokens.reasoning",
            "gen_ai.usage.reasoning.output_tokens",
            _coerce_non_negative_int,
        ),
        SpanType.TOOL: ("tool.call_id", "gen_ai.tool.call.id", _text),
        SpanType.RETRIEVER: (
            "retrieval.top_k",
            "retrieval.top_k",
            _coerce_non_negative_int,
        ),
    }
    if span_type not in mappings:
        return {}
    target, source, convert = mappings[span_type]
    value = fields.take(source, convert=convert)
    return {target: value} if value is not None else {}


_RETRIEVAL_DATA_KEYS = ("input.value", "inputs", "output.value", "outputs")


def _otel_metadata(
    fields: _SourceFields, raw: dict[str, Any], span_type: SpanType
) -> dict[str, Any]:
    """Namespace unconsumed fields; keep retrieval values as received."""
    metadata = fields.remaining("otel")
    if span_type == SpanType.RETRIEVER:
        _restore_raw_retrieval(metadata, raw)
    return metadata


def _restore_raw_retrieval(metadata: dict[str, Any], raw: dict[str, Any]) -> None:
    """Keep retriever inputs/outputs in metadata as received from the source."""
    raw_attributes = _attr_map(raw.get("attributes"), parse_json=False)
    for key in _RETRIEVAL_DATA_KEYS:
        meta_key = f"otel.{key}"
        if meta_key in metadata and key in raw_attributes:
            metadata[meta_key] = raw_attributes[key]


def span_from_otel(raw: dict[str, Any], *, trace_id: str | None = None) -> Span:
    """Convert OTEL attributes, retaining values not successfully normalized."""
    fields = _SourceFields(_attr_map(raw.get("attributes")))
    resolved_trace_id, span_id = _require_ids(
        trace_id=_correlated_trace_id(
            trace_id, raw.get("traceId"), raw.get("trace_id"), fields.take("trace_id")
        ),
        span_id=raw.get("spanId") or raw.get("span_id"),
        missing_trace="OTEL span is missing traceId",
        missing_span="OTEL span is missing spanId",
    )
    start_time = _timestamp(
        raw, "startTimeUnixNano", "startTime", "start_time", parse=_as_datetime
    )
    end_time = _timestamp(
        raw, "endTimeUnixNano", "endTime", "end_time", parse=_as_datetime
    )
    status, message = _status_from_otel(raw.get("status"))
    parent = _first_present(raw.get("parentSpanId"), raw.get("parent_span_id"))
    span_type = (
        fields.take(
            "mlflow.spanType",
            "gen_ai.operation.name",
            "openinference.span.kind",
            "span.type",
            "langfuse.observation.type",
            convert=_known_span_type,
        )
        or SpanType.UNKNOWN
    )
    name = str(raw.get("name") or span_id)
    llm = _source_llm(
        fields,
        {
            "model": ("gen_ai.request.model", "gen_ai.response.model"),
            "provider": ("gen_ai.system", "gen_ai.provider.name"),
            "temperature": ("gen_ai.request.temperature",),
            "input": ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"),
            "output": (
                "gen_ai.usage.output_tokens",
                "gen_ai.usage.completion_tokens",
            ),
            "total": ("gen_ai.usage.total_tokens",),
            "cost": ("gen_ai.usage.cost", "cost_usd"),
        },
    )
    attributes = _standard_attributes(
        span_type=span_type,
        name=(
            fields.take("gen_ai.tool.name", convert=_text) or name
            if span_type == SpanType.TOOL
            else name
        ),
        data=_semantic_data(
            fields,
            span_type,
            (
                ("gen_ai.prompt", "input.value", "inputs")
                if span_type == SpanType.GENERATION
                else ("input.value", "inputs")
            ),
            (
                ("gen_ai.completion", "output.value", "outputs")
                if span_type == SpanType.GENERATION
                else ("output.value", "outputs")
            ),
        ),
        llm=llm,
        status=(status, message),
    )
    attributes.update(_otel_type_attributes(fields, span_type))
    tags = _source_tags(raw, fields)
    return Span(
        span_id=span_id,
        trace_id=resolved_trace_id,
        parent_span_id=str(parent) if parent else None,
        source=TracePlatform.OTEL,
        name=name,
        span_type=span_type,
        status=status,
        status_message=message,
        start_time=start_time,
        end_time=end_time,
        latency=_source_latency(fields, start_time, end_time),
        attributes=attributes,
        metadata=_otel_metadata(fields, raw, span_type),
        events=_span_events(raw.get("events")),
        tags=tags,
        llm=llm,
    )


def _otel_spans_from_payload(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract span dicts from flat or resourceSpans OTEL payloads."""
    if isinstance(raw.get("spans"), list):
        return _payload_spans(raw["spans"])
    spans_raw: list[dict[str, Any]] = []
    for resource_span in raw.get("resourceSpans") or raw.get("resource_spans") or []:
        for scope_span in (
            resource_span.get("scopeSpans") or resource_span.get("scope_spans") or []
        ):
            spans_raw.extend(_payload_spans(scope_span.get("spans") or []))
    return spans_raw


def _span_attribute(spans_raw: list[dict[str, Any]], key: str) -> Any:
    """Read a span attribute from the first span that carries a value."""
    for item in spans_raw:
        value = _attr_map(item.get("attributes")).get(key)
        if value is not None:
            return value
    return None


def trace_from_otel(raw: dict[str, Any]) -> Trace:
    """Convert an OTEL resource-spans / span-list payload to :class:`Trace`."""
    spans_raw = _otel_spans_from_payload(raw)
    if not spans_raw:
        raise DataValidationError("OTEL payload contains no spans")

    span_identities = [
        identity
        for item in spans_raw
        for identity in (
            item.get("traceId"),
            item.get("trace_id"),
            _attr_map(item.get("attributes")).get("trace_id"),
        )
        if identity is not None
    ]
    trace_id = _correlated_trace_id(
        _first_present(raw.get("traceId"), raw.get("trace_id")),
        *span_identities,
    )
    if not trace_id:
        raise DataValidationError("OTEL payload is missing traceId")

    spans = [span_from_otel(item, trace_id=str(trace_id)) for item in spans_raw]
    return Trace(
        trace_id=str(trace_id),
        name=raw.get("name"),
        session_id=raw.get("session_id") or _span_attribute(spans_raw, "session.id"),
        platform=TracePlatform.OTEL,
        start_time=min((s.start_time for s in spans if s.start_time), default=None),
        end_time=max((s.end_time for s in spans if s.end_time), default=None),
        metadata=_metadata_extras(raw.get("metadata"), "otel"),
        tags=_tags(raw.get("tags")) or [],
        spans=spans,
    )
