"""Shared semantic span attributes, events, and typed LLM data construction."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from lightspeed_evaluation.core.models.trace import (
    LLMSpanData,
    ModelInfo,
    SpanEvent,
    SpanStatus,
    SpanType,
    TokenUsage,
    normalize_span_type,
)
from lightspeed_evaluation.core.tracing._helpers import (
    _as_datetime,
    _attr_map,
    _coerce_float,
    _coerce_non_negative_float,
    _coerce_non_negative_int,
    _first_present,
    _latency_seconds,
    _parse_jsonish,
    _SourceFields,
    _tags,
    _text,
    _timestamp,
)


def _known_span_type(value: Any) -> Optional[SpanType]:
    """Keep unknown source type labels in metadata rather than consuming them."""
    normalized = normalize_span_type(value)
    return normalized if normalized != SpanType.UNKNOWN else None


_MAPPED_SPAN_TYPES = {
    SpanType.TOOL,
    SpanType.GENERATION,
    SpanType.RETRIEVER,
    SpanType.EVENT,
}


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
        if isinstance(inputs, dict):
            attrs["retrieval.top_k"] = _coerce_non_negative_int(inputs.get("top_k"))
        if outputs is not None:
            attrs.update(_retrieval_attributes(outputs))
    elif span_type == SpanType.EVENT:
        body = _first_present(outputs, inputs)
        attrs = {"event.name": name, "event.body": body}
        if isinstance(body, dict):
            attrs["event.phase"] = body.get("phase")
    return {key: value for key, value in attrs.items() if value is not None}


def _retrieval_attributes(outputs: Any) -> dict[str, Any]:
    """Extract text; emit optional columns only when aligned with every chunk."""
    documents = outputs
    if isinstance(outputs, dict):
        documents = _first_present(
            outputs.get("documents"), outputs.get("chunks"), outputs.get("results")
        )
    if isinstance(documents, str):
        documents = [documents]
    if not isinstance(documents, list):
        return {}
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
        return {}
    attrs: dict[str, Any] = {"retrieval.chunks": [row[0] for row in rows]}
    if rows and all(row[1] is not None for row in rows):
        attrs["retrieval.sources"] = [row[1] for row in rows]
    if rows and all(row[2] is not None for row in rows):
        attrs["retrieval.scores"] = [row[2] for row in rows]
    return attrs


def _source_llm(
    fields: _SourceFields, keys: dict[str, tuple[str, ...]]
) -> Optional[LLMSpanData]:
    """Normalize model and usage while retaining unsuccessful source values."""
    return _llm_span_data(
        model=_model_info(
            fields.take(*keys.get("model", ()), convert=_text),
            fields.take(*keys.get("provider", ()), convert=_text),
            fields.take(*keys.get("temperature", ()), convert=_coerce_float),
        ),
        token_usage=_token_usage(
            fields.take(*keys.get("input", ()), convert=_coerce_non_negative_int),
            fields.take(*keys.get("output", ()), convert=_coerce_non_negative_int),
            fields.take(*keys.get("total", ()), convert=_coerce_non_negative_int),
        ),
        cost=fields.take(*keys.get("cost", ()), convert=_coerce_non_negative_float),
    )


def _source_latency(
    fields: _SourceFields, start: Optional[datetime], end: Optional[datetime]
) -> Optional[float]:
    """Consume only the chosen explicit latency representation."""
    seconds = fields.take("latency", convert=_coerce_non_negative_float)
    if seconds is not None:
        return seconds
    milliseconds = fields.take("latency_ms", convert=_coerce_non_negative_float)
    return _latency_seconds(start, end, explicit_ms=milliseconds)


def _source_tags(raw: dict[str, Any], fields: _SourceFields) -> list[str]:
    """Prefer valid top-level tags; consume attribute tags only when selected."""
    tags = _tags(raw.get("tags"))
    if tags is not None:
        return tags
    return fields.take("tags", convert=_tags) or []


def _semantic_data(
    fields: _SourceFields,
    span_type: SpanType,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> tuple[Any, Any]:
    """Consume lossless input/output mappings; keep structured extras.

    Retrieval parsing can be partial, so retrieval inputs/outputs are never
    consumed — the raw values always remain available in metadata. EVENT has
    a single body, so only the selected input or output is consumed.
    """
    if span_type not in _MAPPED_SPAN_TYPES:
        return None, None
    if span_type == SpanType.RETRIEVER:
        return (
            _first_present(*(fields.values.get(key) for key in inputs)),
            _first_present(*(fields.values.get(key) for key in outputs)),
        )
    if span_type == SpanType.EVENT:
        body = fields.take(*outputs, *inputs)
        return None, body
    return (
        fields.take(*inputs),
        fields.take(*outputs),
    )


def _span_events(raw: Any) -> list[SpanEvent]:
    """Normalize named span events with OTLP or timestamp/attributes fields."""
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
