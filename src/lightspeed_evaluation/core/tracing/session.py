"""Group normalized traces into trace sessions."""

from typing import Any

from lightspeed_evaluation.core.models.trace import Trace, TraceSession


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
