"""Adapters that convert already-fetched platform payloads into internal models.

These helpers assume platform-specific ingestion (API / SDK fetch) is already
done. They only normalize dict-shaped payloads into :class:`Trace` /
:class:`TraceSession`. They are not wired into the evaluation pipeline.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any, Optional

from lightspeed_evaluation.core.models._trace_helpers import (
    _coerce_float,
    _coerce_int,
    _coerce_non_negative_float,
    _coerce_non_negative_int,
    _correlated_trace_id,
    _mlflow_identity,
    _payload_spans,
    _tags,
    _timestamp,
)
from lightspeed_evaluation.core.models.trace import (
    LLMSpanData,
    ModelInfo,
    Span,
    SpanEvent,
    SpanStatus,
    SpanType,
    TokenUsage,
    Trace,
    TracePlatform,
    TraceSession,
    normalize_span_type,
)
from lightspeed_evaluation.core.system.exceptions import DataValidationError

_MLFLOW_META_SKIP = {
    "mlflow.traceRequestId",
    "mlflow.spanType",
    "mlflow.spanInputs",
    "mlflow.spanOutputs",
    "mlflow.spanModel",
    "mlflow.spanCost",
    "llm.usage.prompt_tokens",
    "llm.usage.completion_tokens",
    "llm.usage.total_tokens",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "model",
    "cost_usd",
    "span_type",
    "latency",
    "latency_ms",
    "tags",
}


def _parse_jsonish(value: Any) -> Any:
    """Parse JSON strings; return other values unchanged."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    should_try = text[0] in '{"[' or (
        len(text) >= 2 and text[0] == '"' and text[-1] == '"'
    )
    if not should_try:
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _ensure_utc(value: datetime) -> datetime:
    """Normalize datetimes to timezone-aware UTC.

    Naive values are assumed UTC; aware values are converted to UTC.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _epoch_to_datetime(number: float) -> datetime:
    """Convert epoch seconds/ms/us/ns heuristic to UTC datetime."""
    if number >= 1e16:
        number /= 1e9
    elif number >= 1e14:
        number /= 1e6
    elif number >= 1e12:
        number /= 1e3
    return datetime.fromtimestamp(number, tz=UTC)


def _as_datetime(value: Any) -> Optional[datetime]:
    """Convert common platform time representations to UTC datetime.

    All successful conversions return timezone-aware UTC. Missing timezone on
    datetime objects or ISO-8601 strings is treated as UTC.
    """
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)):
            number = float(value)
            return _epoch_to_datetime(number) if math.isfinite(number) else None
        if isinstance(value, str) and value.strip():
            text = value.strip()
            if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                return _epoch_to_datetime(float(text))
            return _ensure_utc(datetime.fromisoformat(text.replace("Z", "+00:00")))
    except (ValueError, OverflowError, OSError):
        pass
    return None


def _first_present(*values: Any) -> Any:
    """Return the first value that is not None (keeps meaningful zeros)."""
    for value in values:
        if value is not None:
            return value
    return None


def _latency_seconds(
    start: Optional[datetime],
    end: Optional[datetime],
    *,
    explicit_seconds: Any = None,
    explicit_ms: Any = None,
) -> Optional[float]:
    """Normalize platform latency to seconds.

    Preference order: explicit seconds, explicit milliseconds, then start/end.
    Internal :class:`Span.latency` is always in seconds (same unit as
    ``agent_latency`` / ``evaluation_latency`` elsewhere in the framework).
    """
    coerced = _coerce_non_negative_float(explicit_seconds)
    if coerced is not None:
        return coerced
    coerced_ms = _coerce_non_negative_float(explicit_ms)
    if coerced_ms is not None:
        return coerced_ms / 1000.0
    if start is not None and end is not None:
        return max(0.0, (end - start).total_seconds())
    return None


def _token_usage(
    input_tokens: Any = None,
    output_tokens: Any = None,
    total_tokens: Any = None,
) -> Optional[TokenUsage]:
    """Build TokenUsage when any non-negative count coerces successfully."""
    usage = TokenUsage(
        input_tokens=_coerce_non_negative_int(input_tokens),
        output_tokens=_coerce_non_negative_int(output_tokens),
        total_tokens=_coerce_non_negative_int(total_tokens),
    )
    if (
        usage.input_tokens is None
        and usage.output_tokens is None
        and usage.total_tokens is None
    ):
        return None
    return usage


def _model_info(
    name: Any = None, provider: Any = None, temperature: Any = None
) -> Optional[ModelInfo]:
    """Build ModelInfo when any identity field is present."""
    coerced_temperature = _coerce_float(temperature)
    if name is None and provider is None and coerced_temperature is None:
        return None
    return ModelInfo(
        name=str(name) if name is not None else None,
        provider=str(provider) if provider is not None else None,
        temperature=coerced_temperature,
    )


def _llm_span_data(
    *,
    model: Optional[ModelInfo] = None,
    token_usage: Optional[TokenUsage] = None,
    cost: Any = None,
) -> Optional[LLMSpanData]:
    """Build :class:`LLMSpanData` when model, token usage, or cost is present.

    Returns ``None`` when no LLM-specific fields are available. Timing lives on
    :class:`Span.latency` (seconds) for all span types; it is not an LLM-only
    signal.
    """
    cost_usd = _coerce_non_negative_float(cost)
    if model is None and token_usage is None and cost_usd is None:
        return None
    return LLMSpanData(
        model=model,
        token_usage=token_usage,
        cost_usd=cost_usd,
    )


def _status_from_mlflow(code: Any) -> SpanStatus:
    normalized = str(code or "").strip().upper()
    if normalized in {"OK", "STATUS_CODE_OK"}:
        return SpanStatus.OK
    if normalized in {"ERROR", "STATUS_CODE_ERROR"}:
        return SpanStatus.ERROR
    return SpanStatus.UNSET


def _status_from_langfuse(level: Any, status_message: Any = None) -> SpanStatus:
    normalized = str(level or "").strip().upper()
    if normalized == "ERROR":
        return SpanStatus.ERROR
    if status_message or normalized in {"DEFAULT", "DEBUG", "WARNING"}:
        return SpanStatus.OK
    return SpanStatus.UNSET


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


def _otel_array_value(array: Any) -> list[Any]:
    """Unwrap an OTLP arrayValue into a plain list."""
    items = array.get("values") if isinstance(array, dict) else None
    if not isinstance(items, list):
        return []
    return [_otel_attr_value(item) for item in items]


def _otel_kvlist_value(kvlist: Any) -> dict[str, Any]:
    """Unwrap an OTLP kvlistValue into a plain dict."""
    items = kvlist.get("values") if isinstance(kvlist, dict) else None
    if not isinstance(items, list):
        return {}
    result: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("key") is None:
            continue
        result[str(item["key"])] = _otel_attr_value(item.get("value"))
    return result


def _otel_attr_value(value: Any) -> Any:
    """Unwrap an OTEL / OTLP AnyValue object into a plain Python value."""
    if not isinstance(value, dict):
        return _parse_jsonish(value)

    if "stringValue" in value:
        result: Any = _parse_jsonish(value["stringValue"])
    elif "intValue" in value:
        coerced_int = _coerce_int(value["intValue"])
        result = value["intValue"] if coerced_int is None else coerced_int
    elif "doubleValue" in value:
        coerced_float = _coerce_float(value["doubleValue"])
        result = value["doubleValue"] if coerced_float is None else coerced_float
    elif "boolValue" in value:
        result = value["boolValue"]
    elif "bytesValue" in value:
        result = value["bytesValue"]
    elif "arrayValue" in value:
        result = _otel_array_value(value["arrayValue"])
    elif "kvlistValue" in value:
        result = _otel_kvlist_value(value["kvlistValue"])
    else:
        result = _parse_jsonish(value)
    return result


def _attr_map(attributes: Any) -> dict[str, Any]:
    """Normalize OTEL attribute list or dict into a plain dict."""
    if isinstance(attributes, dict):
        return {str(k): _otel_attr_value(v) for k, v in attributes.items()}
    if not isinstance(attributes, list):
        return {}
    result: dict[str, Any] = {}
    for item in attributes:
        if not isinstance(item, dict) or item.get("key") is None:
            continue
        result[str(item["key"])] = _otel_attr_value(
            item.get("value", item.get("stringValue"))
        )
    return result


def _require_ids(
    *,
    trace_id: Any,
    span_id: Any,
    missing_trace: str,
    missing_span: str,
) -> tuple[str, str]:
    """Validate and stringify required span identity fields."""
    if not trace_id:
        raise DataValidationError(missing_trace)
    if not span_id:
        raise DataValidationError(missing_span)
    return str(trace_id), str(span_id)


def _llm_attributes(llm: Optional[LLMSpanData]) -> dict[str, Any]:
    """Derive standard keys from the same typed data used by metrics."""
    if llm is None:
        return {}
    attrs: dict[str, Any] = {"cost.usd": llm.cost_usd}
    if llm.model is not None:
        attrs.update(
            {
                "model.name": llm.model.name,
                "model.provider": llm.model.provider,
                "model.temperature": llm.model.temperature,
            }
        )
    if llm.token_usage is not None:
        attrs.update(
            {
                "tokens.input": llm.token_usage.input_tokens,
                "tokens.output": llm.token_usage.output_tokens,
                "tokens.total": llm.token_usage.total_tokens,
            }
        )
    return attrs


def _standard_attributes(
    *,
    span_type: SpanType,
    name: str,
    data: tuple[Any, Any],
    llm: Optional[LLMSpanData],
    status: tuple[SpanStatus, Optional[str]],
) -> dict[str, Any]:
    """Build semantic keys from input/output and status/message pairs."""
    inputs, outputs = (_parse_jsonish(value) for value in data)
    attrs: dict[str, Any] = {}
    if span_type == SpanType.TOOL:
        attrs = {"tool.name": name, "tool.arguments": inputs, "tool.result": outputs}
        if status[0] == SpanStatus.ERROR:
            attrs["tool.error"] = status[1]
    elif span_type == SpanType.GENERATION:
        attrs = {"generation.prompt": inputs, "generation.completion": outputs}
        attrs.update(_llm_attributes(llm))
    elif span_type == SpanType.RETRIEVER:
        attrs["retrieval.query"] = (
            _first_present(inputs.get("query"), inputs.get("input"))
            if isinstance(inputs, dict)
            else inputs
        )
        if isinstance(inputs, dict) and inputs.get("top_k") is not None:
            attrs["retrieval.top_k"] = inputs["top_k"]
        if outputs is not None:
            attrs.update(_retrieval_attributes(outputs)[0])
    elif span_type == SpanType.EVENT:
        body = _first_present(outputs, inputs)
        attrs = {"event.name": name, "event.body": body}
        if isinstance(body, dict):
            attrs["event.phase"] = body.get("phase")
    return {key: value for key, value in attrs.items() if value is not None}


def _retrieval_attributes(outputs: Any) -> tuple[dict[str, Any], bool]:
    """Normalize document lists; report whether attributes are lossless."""
    documents = outputs
    if isinstance(outputs, dict):
        documents = _first_present(
            outputs.get("documents"), outputs.get("chunks"), outputs.get("results")
        )
    if isinstance(documents, str):
        documents = [documents]
    if not isinstance(documents, list):
        return {}, False
    rows: list[tuple[str, Any, Any]] = []
    for document in documents:
        if isinstance(document, str):
            rows.append((document, None, None))
        elif isinstance(document, dict):
            text = _first_present(
                document.get("page_content"),
                document.get("text"),
                document.get("content"),
            )
            if not isinstance(text, str):
                continue
            metadata = document.get("metadata")
            source = _first_present(
                document.get("source"),
                metadata.get("source") if isinstance(metadata, dict) else None,
            )
            rows.append((text, source, _coerce_float(document.get("score"))))
    if not rows and documents:
        return {}, False
    attrs: dict[str, Any] = {"retrieval.chunks": [row[0] for row in rows]}
    lossless = len(rows) == len(documents)
    for column, key in ((1, "retrieval.sources"), (2, "retrieval.scores")):
        values = [row[column] for row in rows]
        if rows and all(value is not None for value in values):
            attrs[key] = values
        elif any(value is not None for value in values):
            lossless = False
    return attrs, lossless


# Span types with standard attribute mappings in _standard_attributes.
_MAPPED_SPAN_TYPES = {
    SpanType.TOOL,
    SpanType.GENERATION,
    SpanType.RETRIEVER,
    SpanType.EVENT,
}


def _unmapped_to_metadata(
    metadata: dict[str, Any],
    *,
    span_type: SpanType,
    prefix: str,
    inputs: Any,
    outputs: Any,
) -> dict[str, Any]:
    """Keep inputs/outputs of unmapped span types in metadata, namespaced.

    Spec §3.4 rule 6: platform-specific extras that don't map to standard
    attribute keys go to ``metadata`` with a platform namespace prefix.
    """
    result = {
        key if key.startswith(f"{prefix}.") else f"{prefix}.{key}": value
        for key, value in metadata.items()
    }
    if span_type in _MAPPED_SPAN_TYPES:
        parsed_inputs = _parse_jsonish(inputs)
        if span_type == SpanType.EVENT and inputs is not None and outputs is not None:
            # The body prefers the output; keep the input available.
            result[f"{prefix}.inputs"] = inputs
        if (
            span_type == SpanType.RETRIEVER
            and isinstance(parsed_inputs, dict)
            and set(parsed_inputs) - {"query", "input", "top_k"}
        ):
            # Input keys beyond the standard retrieval set are not represented.
            result[f"{prefix}.inputs"] = inputs
        if (
            span_type == SpanType.RETRIEVER
            and outputs is not None
            and not _retrieval_attributes(_parse_jsonish(outputs))[1]
        ):
            result[f"{prefix}.outputs"] = outputs
        return result
    if inputs is not None:
        result[f"{prefix}.inputs"] = inputs
    if outputs is not None:
        result[f"{prefix}.outputs"] = outputs
    return result


def _mlflow_llm(attributes: dict[str, Any]) -> Optional[LLMSpanData]:
    """Extract optional LLM fields from MLflow attributes."""
    return _llm_span_data(
        model=_model_info(
            _first_present(attributes.get("mlflow.spanModel"), attributes.get("model"))
        ),
        token_usage=_token_usage(
            _first_present(
                attributes.get("llm.usage.prompt_tokens"),
                attributes.get("input_tokens"),
            ),
            _first_present(
                attributes.get("llm.usage.completion_tokens"),
                attributes.get("output_tokens"),
            ),
            _first_present(
                attributes.get("llm.usage.total_tokens"),
                attributes.get("total_tokens"),
            ),
        ),
        cost=_first_present(
            attributes.get("cost_usd"), attributes.get("mlflow.spanCost")
        ),
    )


def span_from_mlflow(raw: dict[str, Any], *, trace_id: str | None = None) -> Span:
    """Convert a single MLflow span dict into an internal :class:`Span`."""
    context = raw.get("context") or {}
    attributes = {
        str(k): _parse_jsonish(v) for k, v in (raw.get("attributes") or {}).items()
    }
    metadata = {k: v for k, v in attributes.items() if k not in _MLFLOW_META_SKIP}
    resolved_trace_id, span_id = _require_ids(
        trace_id=_mlflow_identity(raw, context, attributes, trace_id, metadata),
        span_id=context.get("span_id") or raw.get("span_id"),
        missing_trace="MLflow span is missing trace_id",
        missing_span="MLflow span is missing span_id",
    )
    start_time = _as_datetime(raw.get("start_time"))
    end_time = _as_datetime(raw.get("end_time"))
    span_type = normalize_span_type(
        attributes.get("mlflow.spanType") or attributes.get("span_type")
    )
    status = _status_from_mlflow(raw.get("status_code"))
    status_message = raw.get("status_message") or None
    inputs = attributes.get("mlflow.spanInputs", raw.get("inputs"))
    outputs = attributes.get("mlflow.spanOutputs", raw.get("outputs"))
    llm = _mlflow_llm(attributes)
    return Span(
        span_id=span_id,
        trace_id=resolved_trace_id,
        parent_span_id=(
            str(raw["parent_id"]) if raw.get("parent_id") is not None else None
        ),
        source=TracePlatform.MLFLOW,
        name=str(raw.get("name") or span_id),
        span_type=span_type,
        status=status,
        status_message=status_message,
        start_time=start_time,
        end_time=end_time,
        latency=_latency_seconds(
            start_time,
            end_time,
            explicit_seconds=attributes.get("latency"),
            explicit_ms=attributes.get("latency_ms"),
        ),
        attributes=_standard_attributes(
            span_type=span_type,
            name=str(raw.get("name") or span_id),
            data=(inputs, outputs),
            llm=llm,
            status=(status, status_message),
        ),
        metadata=_unmapped_to_metadata(
            metadata,
            span_type=span_type,
            prefix="mlflow",
            inputs=inputs,
            outputs=outputs,
        ),
        events=_otel_events(raw.get("events")),
        tags=_tags(_first_present(raw.get("tags"), attributes.get("tags"))) or [],
        llm=llm,
    )


def trace_from_mlflow(raw: dict[str, Any]) -> Trace:
    """Convert an MLflow trace payload (``{"spans": [...], ...}``) to :class:`Trace`."""
    spans_raw = _payload_spans(
        _first_present(raw.get("spans"), (raw.get("data") or {}).get("spans"), [])
    )

    info = raw.get("info") or {}
    first_span = spans_raw[0] if spans_raw else {}
    first_context = first_span.get("context") or {}
    first_request_id = _parse_jsonish(
        (first_span.get("attributes") or {}).get("mlflow.traceRequestId")
    )
    request_id = _first_present(
        raw.get("trace_id"),
        info.get("request_id"),
        info.get("trace_id"),
        first_request_id,
        first_context.get("trace_id"),
    )
    if request_id is None and spans_raw:
        request_id = span_from_mlflow(spans_raw[0]).trace_id
    if request_id is None:
        raise DataValidationError("MLflow trace payload is missing trace_id")
    request_id = str(request_id)
    if _first_present(
        raw.get("trace_id"),
        info.get("request_id"),
        info.get("trace_id"),
        first_request_id,
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
        trace_id=str(request_id),
        name=info.get("name") or raw.get("name"),
        session_id=(
            tags.get("mlflow.trace.session")
            if isinstance(tags, dict)
            else raw.get("session_id")
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
                **(info.get("metadata") or {}),
            }.items()
            if v is not None
        },
        tags=tag_list,
        spans=[span_from_mlflow(item, trace_id=str(request_id)) for item in spans_raw],
    )


def _langfuse_latency(
    raw: dict[str, Any],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
) -> Optional[float]:
    """Langfuse latency is already reported in seconds."""
    return _latency_seconds(start_time, end_time, explicit_seconds=raw.get("latency"))


def _langfuse_llm(raw: dict[str, Any]) -> Optional[LLMSpanData]:
    """Extract optional LLM fields from a Langfuse observation."""
    usage = _first_present(raw.get("usage"), raw.get("usageDetails")) or {}
    token_usage = None
    if isinstance(usage, dict) and usage:
        token_usage = _token_usage(
            _first_present(usage.get("input"), usage.get("promptTokens")),
            _first_present(usage.get("output"), usage.get("completionTokens")),
            _first_present(usage.get("total"), usage.get("totalTokens")),
        )
    return _llm_span_data(
        model=_model_info(
            _first_present(
                raw.get("model"), (raw.get("modelParameters") or {}).get("model")
            ),
            temperature=(raw.get("modelParameters") or {}).get("temperature"),
        ),
        token_usage=token_usage,
        cost=_first_present(
            raw.get("calculatedTotalCost"), raw.get("totalCost"), raw.get("cost")
        ),
    )


def span_from_langfuse(raw: dict[str, Any], *, trace_id: str | None = None) -> Span:
    """Convert a Langfuse observation dict into an internal :class:`Span`."""
    resolved_trace_id, span_id = _require_ids(
        trace_id=_correlated_trace_id(
            trace_id, raw.get("traceId"), raw.get("trace_id")
        ),
        span_id=raw.get("id") or raw.get("observationId") or raw.get("span_id"),
        missing_trace="Langfuse observation is missing traceId",
        missing_span="Langfuse observation is missing id",
    )
    start_time = _as_datetime(
        _first_present(raw.get("startTime"), raw.get("start_time"))
    )
    end_time = _as_datetime(_first_present(raw.get("endTime"), raw.get("end_time")))
    parent = raw.get("parentObservationId") or raw.get("parent_observation_id")
    span_type = normalize_span_type(raw.get("type") or raw.get("observationType"))
    status = _status_from_langfuse(raw.get("level"), raw.get("statusMessage"))
    status_message = raw.get("statusMessage") or raw.get("status_message")
    name = str(raw.get("name") or span_id)
    tool_name = str(
        raw.get("name") or (raw.get("metadata") or {}).get("tool_name") or span_id
    )
    inputs = raw.get("input") if "input" in raw else raw.get("inputs")
    outputs = raw.get("output") if "output" in raw else raw.get("outputs")
    llm = _langfuse_llm(raw)
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
            name=tool_name if span_type == SpanType.TOOL else name,
            data=(inputs, outputs),
            llm=llm,
            status=(status, status_message),
        ),
        metadata=_unmapped_to_metadata(
            {k: v for k, v in (raw.get("metadata") or {}).items() if k != "tool_name"},
            span_type=span_type,
            prefix="langfuse",
            inputs=inputs,
            outputs=outputs,
        ),
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
            _first_present(raw.get("timestamp"), raw.get("startTime"))
        ),
        end_time=_as_datetime(raw.get("endTime")),
        metadata=dict(raw.get("metadata") or {}),
        tags=_tags(raw.get("tags")) or [],
        spans=spans,
    )


def _otel_span_type(attributes: dict[str, Any]) -> Any:
    """Resolve a raw span type from common OTEL attribute keys."""
    return (
        attributes.get("mlflow.spanType")
        or attributes.get("gen_ai.operation.name")
        or attributes.get("openinference.span.kind")
        or attributes.get("span.type")
        or attributes.get("langfuse.observation.type")
    )


def _otel_llm(attributes: dict[str, Any]) -> Optional[LLMSpanData]:
    """Extract optional LLM fields from OTEL / GenAI attributes."""
    return _llm_span_data(
        model=_model_info(
            _first_present(
                attributes.get("gen_ai.request.model"),
                attributes.get("gen_ai.response.model"),
            ),
            _first_present(
                attributes.get("gen_ai.system"),
                attributes.get("gen_ai.provider.name"),
            ),
            attributes.get("gen_ai.request.temperature"),
        ),
        token_usage=_token_usage(
            _first_present(
                attributes.get("gen_ai.usage.input_tokens"),
                attributes.get("gen_ai.usage.prompt_tokens"),
            ),
            _first_present(
                attributes.get("gen_ai.usage.output_tokens"),
                attributes.get("gen_ai.usage.completion_tokens"),
            ),
            attributes.get("gen_ai.usage.total_tokens"),
        ),
        cost=_first_present(
            attributes.get("gen_ai.usage.cost"), attributes.get("cost_usd")
        ),
    )


# OTEL attribute keys consumed into standard span fields / llm data.
_OTEL_CONSUMED_ATTRS = {
    "gen_ai.prompt",
    "gen_ai.completion",
    "input.value",
    "output.value",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.system",
    "gen_ai.provider.name",
    "gen_ai.request.temperature",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "gen_ai.usage.total_tokens",
    "gen_ai.usage.cost",
    "cost_usd",
    "gen_ai.tool.name",
    "gen_ai.tool.call.id",
    "gen_ai.usage.reasoning.output_tokens",
    "retrieval.top_k",
}


def _otel_semantic_extras(
    attributes: dict[str, Any], span_type: SpanType
) -> dict[str, Any]:
    """Map semantic fields without an LLM convenience counterpart."""
    mappings = {
        SpanType.GENERATION: {
            "gen_ai.usage.reasoning.output_tokens": "tokens.reasoning",
        },
        SpanType.TOOL: {"gen_ai.tool.call.id": "tool.call_id"},
        SpanType.RETRIEVER: {"retrieval.top_k": "retrieval.top_k"},
    }
    result = {}
    for source, target in mappings.get(span_type, {}).items():
        value = attributes.get(source)
        if target == "tokens.reasoning":
            value = _coerce_non_negative_int(value)
        if value is not None:
            result[target] = value
    return result


def _otel_events(raw: Any) -> list[SpanEvent]:
    """Convert OTEL span events into :class:`SpanEvent` instances."""
    events: list[SpanEvent] = []
    for item in raw or []:
        if not isinstance(item, dict) or item.get("name") is None:
            continue
        events.append(
            SpanEvent(
                name=str(item["name"]),
                timestamp=_timestamp(
                    item, "timeUnixNano", "time", "timestamp", parse=_as_datetime
                ),
                attributes=_attr_map(item.get("attributes")),
            )
        )
    return events


def span_from_otel(raw: dict[str, Any], *, trace_id: str | None = None) -> Span:
    """Convert an OpenTelemetry span dict into an internal :class:`Span`."""
    attributes = _attr_map(raw.get("attributes"))
    resolved_trace_id, span_id = _require_ids(
        trace_id=_correlated_trace_id(
            trace_id,
            raw.get("traceId"),
            raw.get("trace_id"),
            attributes.get("trace_id"),
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
    status, status_message = _status_from_otel(raw.get("status"))
    parent = raw.get("parentSpanId") or raw.get("parent_span_id")
    span_type = normalize_span_type(_otel_span_type(attributes))
    name = str(raw.get("name") or span_id)
    inputs = _first_present(
        attributes.get("gen_ai.prompt"),
        attributes.get("input.value"),
        attributes.get("inputs"),
    )
    outputs = _first_present(
        attributes.get("gen_ai.completion"),
        attributes.get("output.value"),
        attributes.get("outputs"),
    )
    llm = _otel_llm(attributes)
    return Span(
        span_id=span_id,
        trace_id=resolved_trace_id,
        parent_span_id=str(parent) if parent else None,
        source=TracePlatform.OTEL,
        name=name,
        span_type=span_type,
        status=status,
        status_message=status_message,
        start_time=start_time,
        end_time=end_time,
        latency=_latency_seconds(
            start_time,
            end_time,
            explicit_seconds=attributes.get("latency"),
            explicit_ms=attributes.get("latency_ms"),
        ),
        attributes=_standard_attributes(
            span_type=span_type,
            name=(
                str(attributes.get("gen_ai.tool.name") or name)
                if span_type == SpanType.TOOL
                else name
            ),
            data=(inputs, outputs),
            llm=llm,
            status=(status, status_message),
        )
        | _otel_semantic_extras(attributes, span_type),
        metadata=_unmapped_to_metadata(
            {k: v for k, v in attributes.items() if k not in _OTEL_CONSUMED_ATTRS},
            span_type=span_type,
            prefix="otel",
            inputs=inputs,
            outputs=outputs,
        ),
        events=_otel_events(raw.get("events")),
        tags=_tags(_first_present(raw.get("tags"), attributes.get("tags"))) or [],
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


def trace_from_otel(raw: dict[str, Any]) -> Trace:
    """Convert an OTEL resource-spans / span-list payload to :class:`Trace`."""
    spans_raw = _otel_spans_from_payload(raw)
    if not spans_raw:
        raise DataValidationError("OTEL payload contains no spans")

    trace_id = _correlated_trace_id(
        _first_present(raw.get("traceId"), raw.get("trace_id")),
        spans_raw[0].get("traceId"),
        spans_raw[0].get("trace_id"),
        _attr_map(spans_raw[0].get("attributes")).get("trace_id"),
    )
    if not trace_id:
        raise DataValidationError("OTEL payload is missing traceId")

    spans = [span_from_otel(item, trace_id=str(trace_id)) for item in spans_raw]
    return Trace(
        trace_id=str(trace_id),
        name=raw.get("name"),
        session_id=raw.get("session_id")
        or _attr_map(spans_raw[0].get("attributes")).get("session.id"),
        platform=TracePlatform.OTEL,
        start_time=min((s.start_time for s in spans if s.start_time), default=None),
        end_time=max((s.end_time for s in spans if s.end_time), default=None),
        metadata=dict(raw.get("metadata") or {}),
        tags=_tags(raw.get("tags")) or [],
        spans=spans,
    )


def session_from_traces(
    session_id: str,
    traces: list[Trace],
    *,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> TraceSession:
    """Group already-normalized traces into a :class:`TraceSession`."""
    return TraceSession(
        session_id=session_id,
        traces=traces,
        metadata=metadata or {},
        tags=tags or [],
    )
