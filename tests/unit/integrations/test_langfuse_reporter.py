"""Tests for optional Langfuse reporter (mocked SDK)."""

from __future__ import annotations

from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import EvaluationResult, EvaluationRunContext
from lightspeed_evaluation.integrations import langfuse_reporter
from lightspeed_evaluation.integrations.langfuse_reporter import (
    _format_comment,
    _score_name,
    build_langfuse_on_complete_callback,
    push_evaluation_results_to_langfuse,
)


def _one_result() -> EvaluationResult:
    return EvaluationResult(
        conversation_group_id="g1",
        turn_id="t1",
        tag="eval",
        metric_identifier="custom:x",
        score=0.9,
        result="PASS",
        reason="ok",
    )


def _install_fake_langfuse(mocker: MockerFixture) -> tuple[Any, Any]:
    """Patch reporter module to use a fake ``Langfuse`` client."""
    mock_lf = mocker.Mock()
    mock_trace = mocker.Mock()
    mock_lf.trace.return_value = mock_trace

    mocker.patch.object(langfuse_reporter, "Langfuse", lambda **kwargs: mock_lf)
    return mock_lf, mock_trace


def test_build_callback_invokes_push(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Build callback and ensure push path is exercised (mocked Langfuse)."""
    mock_lf, mock_trace = _install_fake_langfuse(mocker)

    cb = build_langfuse_on_complete_callback()
    results = [_one_result()]
    ctx = EvaluationRunContext(run_name="r1", original_data_path="/a.yaml")
    caplog.set_level("ERROR")
    cb(results, ctx)

    mock_lf.trace.assert_called_once()
    assert mock_trace.score.call_count == 1
    mock_lf.flush.assert_called_once()


def test_push_with_mock_langfuse(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    """Direct push with mocked client."""
    mock_lf, _ = _install_fake_langfuse(mocker)

    caplog.set_level("ERROR")
    push_evaluation_results_to_langfuse(
        [_one_result()], EvaluationRunContext(run_name="n")
    )
    trace_kwargs = mock_lf.trace.call_args.kwargs
    assert "rows_preview" in trace_kwargs["metadata"]
    preview = trace_kwargs["metadata"]["rows_preview"][0]
    assert preview["idx"] == 0
    assert preview["row_key"] == "0000_g1"
    assert preview["conversation_group_id"] == "g1"
    assert preview["turn_id"] == "t1"
    assert preview["query"] == ""
    assert preview["response"] == ""
    assert "tags" not in trace_kwargs
    assert mock_lf.flush.called


def test_push_skips_none_scores(mocker: MockerFixture) -> None:
    """Rows without numeric score are skipped (no fake 0.0 in Langfuse)."""
    mock_lf, mock_trace = _install_fake_langfuse(mocker)
    scored = _one_result()
    unscored = _one_result().model_copy(update={"score": None, "result": "SKIPPED"})

    push_evaluation_results_to_langfuse(
        [scored, unscored], EvaluationRunContext(run_name="n")
    )
    trace_meta = mock_lf.trace.call_args.kwargs["metadata"]["rows_preview"]
    assert len(trace_meta) == 2
    assert trace_meta[1]["idx"] == 1
    assert trace_meta[1]["row_key"] == "0001_g1"

    assert mock_trace.score.call_count == 1
    kwargs = mock_trace.score.call_args.kwargs
    assert kwargs["value"] == 0.9
    assert kwargs["name"] == "custom:x"
    assert "id" not in kwargs
    meta = kwargs["metadata"]
    assert set(meta.keys()) == {
        "query",
        "response",
        "conversation_group_id",
        "turn_id",
        "tool_calls",
        "contexts",
        "expected_response",
        "expected_intent",
        "expected_tool_calls",
        "expected_keywords",
    }
    assert meta["query"] == ""
    assert meta["response"] == ""
    assert meta["conversation_group_id"] == "g1"
    assert meta["turn_id"] == "t1"
    for key in (
        "tool_calls",
        "contexts",
        "expected_response",
        "expected_intent",
        "expected_tool_calls",
        "expected_keywords",
    ):
        assert meta[key] == ""
    assert mock_lf.flush.called


def test_push_score_metadata_includes_query_response(mocker: MockerFixture) -> None:
    """Per-score metadata includes eval fields for Langfuse Scores UI."""
    _, mock_trace = _install_fake_langfuse(mocker)
    r = _one_result().model_copy(
        update={
            "query": "Why?",
            "response": "Because.",
            "tool_calls": '[{"tool_name": "t"}]',
            "contexts": "ctx1",
            "expected_response": "exp",
            "expected_intent": "intent",
            "expected_tool_calls": "exp_tools",
            "expected_keywords": "kw",
        }
    )
    push_evaluation_results_to_langfuse([r], EvaluationRunContext(run_name="n"))
    meta = mock_trace.score.call_args.kwargs["metadata"]
    assert meta["query"] == "Why?"
    assert meta["response"] == "Because."
    assert meta["tool_calls"] == '[{"tool_name": "t"}]'
    assert meta["contexts"] == "ctx1"
    assert meta["expected_response"] == "exp"
    assert meta["expected_intent"] == "intent"
    assert meta["expected_tool_calls"] == "exp_tools"
    assert meta["expected_keywords"] == "kw"


def test_push_score_metadata_expected_response_list(mocker: MockerFixture) -> None:
    """expected_response list is flattened into metadata string."""
    _, mock_trace = _install_fake_langfuse(mocker)
    r = _one_result().model_copy(update={"expected_response": ["a", "b"]})
    push_evaluation_results_to_langfuse([r], EvaluationRunContext(run_name="n"))
    meta = mock_trace.score.call_args.kwargs["metadata"]
    assert meta["expected_response"] == "a\n---\nb"


def test_push_no_results() -> None:
    """No Langfuse client when there are no rows."""
    push_evaluation_results_to_langfuse([], EvaluationRunContext(run_name="n"))


def test_format_comment() -> None:
    """Format comment helper."""
    r = _one_result()
    c = _format_comment(r)
    assert "result=PASS" in c
    assert "conversation_group_id=g1" in c
    assert "turn_id=t1" in c
    assert "query=" not in c
    assert "response=" not in c
    assert "ok" in c


def test_score_name_sanitizes() -> None:
    """Score name uses metric identifier only."""
    r = _one_result()
    r = r.model_copy(update={"metric_identifier": "a:b"})
    name = _score_name(r)
    assert name == "a:b"
    assert len(name) <= 200
