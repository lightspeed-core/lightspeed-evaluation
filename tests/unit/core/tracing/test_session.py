"""Tests for trace session grouping."""

from lightspeed_evaluation.core.models.trace import Trace, TracePlatform
from lightspeed_evaluation.core.tracing.session import session_from_traces


def test_session_from_traces() -> None:
    """Group multiple traces under one session."""
    t1 = Trace(trace_id="a", platform=TracePlatform.MLFLOW, spans=[])
    t2 = Trace(trace_id="b", platform=TracePlatform.LANGFUSE, spans=[])
    session = session_from_traces("sess-x", [t1, t2], tags=["nightly"])
    assert session.session_id == "sess-x"
    assert len(session.traces) == 2
    assert session.tags == ["nightly"]
    assert all(t.session_id == "sess-x" for t in session.traces)
