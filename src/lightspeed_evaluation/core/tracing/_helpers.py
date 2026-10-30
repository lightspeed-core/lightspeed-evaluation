"""Payload decoding, value coercion, identity validation, and field tracking.

These helpers interpret source values without constructing semantic span data.
Fields are consumed only after successful conversion; remaining source values
stay available for namespaced metadata.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Optional

from lightspeed_evaluation.core.system.exceptions import DataValidationError


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


def _text(value: Any) -> Optional[str]:
    """Accept nonblank text without inventing identities from other objects."""
    return value if isinstance(value, str) and value.strip() else None


class _SourceFields:
    """Read source values, consuming only the selected successful conversion."""

    def __init__(self, values: dict[str, Any]):
        self.values = values
        self.consumed: set[str] = set()

    def take(
        self, *keys: str, convert: Callable[[Any], Any] = lambda value: value
    ) -> Any:
        """Return first valid candidate; retain invalid and unused aliases."""
        for key in keys:
            value = self.values.get(key)
            if value is None:
                continue
            normalized = convert(value)
            if normalized is not None:
                self.consumed.add(key)
                return normalized
        return None

    def nested(self, *keys: str) -> tuple[Optional[str], "_SourceFields"]:
        """Select a dictionary without consuming its unknown members."""
        for key in keys:
            if isinstance(self.values.get(key), dict):
                return key, _SourceFields(dict(self.values[key]))
        return None, _SourceFields({})

    def finish_nested(self, key: Optional[str], nested: "_SourceFields") -> None:
        """Preserve unconsumed nested values; consume fully mapped containers."""
        if key is None or not nested.consumed:
            return
        remaining = {k: v for k, v in nested.values.items() if k not in nested.consumed}
        if remaining:
            self.values[key] = remaining
        else:
            self.consumed.add(key)

    def remaining(self, prefix: str) -> dict[str, Any]:
        """Return unconsumed fields under their source namespace."""
        return {
            key if key.startswith(f"{prefix}.") else f"{prefix}.{key}": value
            for key, value in self.values.items()
            if key not in self.consumed
        }


def _coerce_int(value: Any) -> Optional[int]:
    """Convert finite integral values without truncating fractions or booleans."""
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            return None
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    """Convert to int when possible; drop negatives (model fields use ge=0)."""
    number = _coerce_int(value)
    if number is None or number < 0:
        return None
    return number


def _coerce_float(value: Any) -> Optional[float]:
    """Convert to float when possible; reject non-finite values."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _coerce_non_negative_float(value: Any) -> Optional[float]:
    """Convert to float when possible; drop negatives (model fields use ge=0)."""
    number = _coerce_float(value)
    if number is None or number < 0.0:
        return None
    return number


def _timestamp(
    raw: dict[str, Any], *keys: str, parse: Callable[[Any], Optional[datetime]]
) -> Optional[datetime]:
    """Resolve a timestamp, honoring explicit nanosecond units and zero values."""
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        if key.endswith("UnixNano"):
            number = _coerce_float(value)
            if number is not None:
                try:
                    return datetime.fromtimestamp(number / 1e9, tz=UTC)
                except (ValueError, OverflowError, OSError):
                    pass
        else:
            result = parse(value)
            if result is not None:
                return result
    return None


def _tags(value: Any) -> Optional[list[str]]:
    """Accept a single tag or a list of tag strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


def _correlated_trace_id(expected: Optional[str], *candidates: Any) -> Optional[str]:
    """Fill a missing trace ID, but reject conflicting source identities."""
    identities = [str(value) for value in candidates if value is not None]
    resolved = expected if expected is not None else next(iter(identities), None)
    if any(value != resolved for value in identities):
        raise DataValidationError("Span trace_id conflicts with enclosing trace")
    return resolved


def _payload_spans(value: Any) -> list[dict[str, Any]]:
    """Reject malformed span containers rather than silently losing records."""
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise DataValidationError("Trace payload must contain a list of span objects")
    return value


def _metadata_extras(value: Any, prefix: str) -> dict[str, Any]:
    """Accept mapping metadata; preserve malformed values under a namespace."""
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    return {f"{prefix}.metadata": value}


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
    datetime objects or ISO-8601 strings is treated as UTC. Invalid,
    non-finite, or out-of-range values return ``None``.
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


def _otel_array_value(array: Any, *, parse_json: bool = True) -> list[Any]:
    """Unwrap an OTLP arrayValue into a plain list."""
    items = array.get("values") if isinstance(array, dict) else None
    if not isinstance(items, list):
        return []
    return [_otel_attr_value(item, parse_json=parse_json) for item in items]


def _otel_kvlist_value(kvlist: Any, *, parse_json: bool = True) -> dict[str, Any]:
    """Unwrap an OTLP kvlistValue into a plain dict."""
    items = kvlist.get("values") if isinstance(kvlist, dict) else None
    if not isinstance(items, list):
        return {}
    result: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("key") is None:
            continue
        result[str(item["key"])] = _otel_attr_value(
            item.get("value"), parse_json=parse_json
        )
    return result


def _otel_attr_value(value: Any, *, parse_json: bool = True) -> Any:
    """Unwrap an OTEL / OTLP AnyValue object into a plain Python value.

    ``parse_json=False`` keeps stringValue data as received, so raw values
    can be preserved in metadata while parsing happens at consumption.
    """
    if not isinstance(value, dict):
        return _parse_jsonish(value) if parse_json else value

    if "stringValue" in value:
        result: Any = (
            _parse_jsonish(value["stringValue"]) if parse_json else value["stringValue"]
        )
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
        result = _otel_array_value(value["arrayValue"], parse_json=parse_json)
    elif "kvlistValue" in value:
        result = _otel_kvlist_value(value["kvlistValue"], parse_json=parse_json)
    else:
        result = _parse_jsonish(value) if parse_json else value
    return result


def _attr_map(attributes: Any, *, parse_json: bool = True) -> dict[str, Any]:
    """Normalize OTEL attribute list or dict into a plain dict."""
    if isinstance(attributes, dict):
        return {
            str(k): _otel_attr_value(v, parse_json=parse_json)
            for k, v in attributes.items()
        }
    if not isinstance(attributes, list):
        return {}
    result: dict[str, Any] = {}
    for item in attributes:
        if not isinstance(item, dict) or item.get("key") is None:
            continue
        result[str(item["key"])] = _otel_attr_value(
            item.get("value", item.get("stringValue")), parse_json=parse_json
        )
    return result
