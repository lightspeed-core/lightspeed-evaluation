"""Value normalization helpers shared by the trace platform adapters.

New-code module for the trace normalization fixes, plus the pure numeric
coercion helpers they build on. Relocating the remaining adapter helpers
here (and out of the models package) is planned follow-up refactoring; the
models package should ultimately retain data models and validation only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Optional

from lightspeed_evaluation.core.system.exceptions import DataValidationError


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


def _mlflow_identity(
    raw: dict[str, Any],
    context: dict[str, Any],
    attributes: dict[str, Any],
    trace_id: Optional[str],
    metadata: dict[str, Any],
) -> Any:
    """Resolve MLflow trace identity, preserving a divergent OTel context ID.

    The MLflow request ID is the normalized trace identity. The OTel context
    trace ID is a distinct identifier: a fallback when no request ID exists,
    preserved in metadata when both are present and they differ.
    """
    request_identity = _correlated_trace_id(
        trace_id, attributes.get("mlflow.traceRequestId"), raw.get("trace_id")
    )
    context_trace_id = context.get("trace_id")
    if (
        request_identity is not None
        and context_trace_id is not None
        and str(context_trace_id) != request_identity
    ):
        metadata["mlflow.span_context_trace_id"] = context_trace_id
    return request_identity if request_identity is not None else context_trace_id


def _payload_spans(value: Any) -> list[dict[str, Any]]:
    """Reject malformed span containers rather than silently losing records."""
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise DataValidationError("Trace payload must contain a list of span objects")
    return value
