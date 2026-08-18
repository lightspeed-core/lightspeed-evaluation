"""Tests for custom:loop_eval loop detection."""

from datetime import UTC, datetime
from typing import Any

import pytest

from lightspeed_evaluation.core.metrics.custom.loop_eval import (
    LoopEvalConfig,
    config_from_metadata,
    evaluate_loop_calls,
    evaluate_loops,
    evaluate_loops_from_trace,
)
from lightspeed_evaluation.core.models import EvaluationData, TurnData
from lightspeed_evaluation.core.models.trace import Span, SpanType, Trace
from lightspeed_evaluation.core.system.exceptions import ConfigurationError


def _call(name: str, arguments: dict | None = None, **extra: object) -> dict:
    """Build a tool_calls dict."""
    payload: dict = {"tool_name": name, "arguments": arguments or {}}
    payload.update(extra)
    return payload


def _sequence(*calls: dict) -> list[list[dict]]:
    """Wrap each call as its own sequence, matching API flattening."""
    return [[call] for call in calls]


class TestEvaluateLoopCalls:
    """Tests for evaluate_loop_calls on TurnData.tool_calls sequences."""

    def test_none_tool_calls_is_error(self) -> None:
        """Missing tool_calls returns no score."""
        score, reason = evaluate_loop_calls(None)
        assert score is None
        assert "No tool_calls" in reason

    def test_empty_tool_calls_scores_perfect(self) -> None:
        """Empty list means no loops."""
        score, reason = evaluate_loop_calls([])
        assert score == 1.0
        assert "No loops detected" in reason

    def test_varied_tools_no_loop(self) -> None:
        """Distinct consecutive tools do not loop."""
        tool_calls = _sequence(
            _call("oc_get", {"kind": "pod"}),
            _call("oc_describe", {"kind": "pod"}),
            _call("oc_get", {"kind": "namespace"}),
        )
        score, reason = evaluate_loop_calls(tool_calls)
        assert score == 1.0
        assert "No loops detected" in reason
        assert "3 tool call" in reason

    def test_exact_loop_identical_args(self) -> None:
        """Three consecutive identical calls are an exact loop."""
        repeated = _call("oc_get", {"kind": "pod", "namespace": "default"})
        tool_calls = _sequence(repeated, repeated, repeated)
        score, reason = evaluate_loop_calls(tool_calls)
        assert score is not None
        assert score < 1.0
        assert "Exact loop" in reason
        assert "oc_get" in reason
        assert "tool call index 0" in reason

    def test_two_identical_calls_below_default_threshold(self) -> None:
        """Two identical calls do not meet the default threshold of 3."""
        repeated = _call("search", {"q": "pods"})
        score, reason = evaluate_loop_calls(_sequence(repeated, repeated))
        assert score == 1.0
        assert "No loops detected" in reason

    def test_soft_loop_same_tool_different_args(self) -> None:
        """Same tool with different args is a soft loop, not exact."""
        tool_calls = _sequence(
            _call("search", {"q": "a"}),
            _call("search", {"q": "b"}),
            _call("search", {"q": "c"}),
        )
        score, reason = evaluate_loop_calls(tool_calls)
        assert score is not None
        assert score < 1.0
        assert "Soft loop" in reason
        assert "Exact loop" not in reason
        assert "search" in reason

    def test_exact_loop_does_not_also_report_identical_soft(self) -> None:
        """An exact run is not double-reported as a soft loop of the same window."""
        repeated = _call("oc_get", {"kind": "pod"})
        score, reason = evaluate_loop_calls(_sequence(repeated, repeated, repeated))
        assert "Exact loop" in reason
        assert "Soft loop" not in reason
        assert score is not None

    def test_soft_loop_extends_beyond_exact_run(self) -> None:
        """Identical prefix plus extra same-tool calls reports both kinds."""
        same = _call("search", {"q": "a"})
        tool_calls = _sequence(
            same,
            same,
            same,
            _call("search", {"q": "b"}),
        )
        score, reason = evaluate_loop_calls(tool_calls)
        assert "Exact loop" in reason
        assert "Soft loop" in reason
        assert score is not None
        assert score < 1.0

    def test_configurable_exact_threshold(self) -> None:
        """A lower exact threshold flags shorter identical runs."""
        repeated = _call("oc_get", {"kind": "pod"})
        config = LoopEvalConfig(exact_loop_threshold=2, soft_loop_threshold=5)
        score, reason = evaluate_loop_calls(
            _sequence(repeated, repeated), config=config
        )
        assert score is not None
        assert score < 1.0
        assert "Exact loop" in reason
        assert "threshold 2" in reason

    def test_severity_increases_with_run_length(self) -> None:
        """Longer exact runs score worse than the minimum threshold run."""
        repeated = _call("oc_get", {"kind": "pod"})
        mild, _ = evaluate_loop_calls(_sequence(repeated, repeated, repeated))
        severe, _ = evaluate_loop_calls(
            _sequence(repeated, repeated, repeated, repeated, repeated)
        )
        assert mild is not None
        assert severe is not None
        assert severe < mild
        assert severe == 0.0

    def test_recursive_depth_from_parent_ids(self) -> None:
        """A parent chain deeper than max_recursive_depth is reported."""
        tool_calls = _sequence(
            _call("root", {}, span_id="s1"),
            _call("child", {}, span_id="s2", parent_span_id="s1"),
            _call("leaf", {}, span_id="s3", parent_span_id="s2"),
        )
        config = LoopEvalConfig(
            exact_loop_threshold=10,
            soft_loop_threshold=10,
            max_recursive_depth=2,
        )
        score, reason = evaluate_loop_calls(tool_calls, config=config)
        assert score is not None
        assert score < 1.0
        assert "Excessive recursive depth" in reason
        assert "root -> child -> leaf" in reason

    def test_flat_calls_without_parent_ids_have_no_depth_finding(self) -> None:
        """No span ids means recursive depth is not assessed."""
        tool_calls = _sequence(
            _call("a", {"x": 1}),
            _call("b", {"x": 2}),
            _call("c", {"x": 3}),
        )
        config = LoopEvalConfig(max_recursive_depth=1)
        score, reason = evaluate_loop_calls(tool_calls, config=config)
        assert score == 1.0
        assert "Excessive recursive depth" not in reason

    def test_json_string_arguments_canonicalize(self) -> None:
        """JSON argument strings with different key order still exact-match."""
        tool_calls = _sequence(
            _call("search", None) | {"arguments": '{"b": 2, "a": 1}'},
            {"tool_name": "search", "arguments": '{"a": 1, "b": 2}'},
            {"tool_name": "search", "arguments": {"a": 1, "b": 2}},
        )
        score, reason = evaluate_loop_calls(tool_calls)
        assert score is not None
        assert score < 1.0
        assert "Exact loop" in reason

    def test_reports_turn_id_in_location(self) -> None:
        """Start location includes turn_id when provided."""
        repeated = _call("oc_get", {"kind": "pod"})
        _score, reason = evaluate_loop_calls(
            _sequence(repeated, repeated, repeated), turn_id="t9"
        )
        assert "turn t9, tool call index 0" in reason


class TestEvaluateLoopsTurnAndConversation:
    """Tests for the CustomMetrics-compatible evaluate_loops wrapper."""

    def test_turn_level_requires_turn_data(self) -> None:
        """Turn-level evaluation without TurnData is an error."""
        score, reason = evaluate_loops(None, 0, None, False)
        assert score is None
        assert "TurnData is required" in reason

    def test_turn_level_uses_turn_tool_calls(self) -> None:
        """Turn-level evaluation reads tool_calls from TurnData."""
        turn = TurnData(
            turn_id="t1",
            query="list pods",
            tool_calls=_sequence(
                _call("oc_get", {"kind": "pod"}),
                _call("oc_describe", {"kind": "pod"}),
            ),
        )
        score, reason = evaluate_loops(None, 0, turn, False)
        assert score == 1.0
        assert "No loops detected" in reason

    def test_conversation_level_does_not_join_turns(self) -> None:
        """Repeating a tool on a later query is not a cross-turn loop."""
        repeated = _call("search", {"q": "same"})
        conv = EvaluationData(
            conversation_group_id="c1",
            turns=[
                TurnData(
                    turn_id="t1",
                    query="first",
                    tool_calls=_sequence(repeated, repeated),
                ),
                TurnData(
                    turn_id="t2",
                    query="second",
                    tool_calls=_sequence(repeated),
                ),
            ],
        )
        score, reason = evaluate_loops(conv, None, None, True)
        assert score == 1.0
        assert "No loops detected" in reason
        assert "Exact loop" not in reason

    def test_conversation_level_detects_in_turn_loop(self) -> None:
        """A loop fully inside one turn still fails conversation-level eval."""
        repeated = _call("search", {"q": "same"})
        conv = EvaluationData(
            conversation_group_id="c1",
            turns=[
                TurnData(
                    turn_id="t1",
                    query="first",
                    tool_calls=_sequence(repeated, repeated, repeated),
                ),
                TurnData(
                    turn_id="t2",
                    query="second",
                    tool_calls=_sequence(_call("oc_get", {"kind": "pod"})),
                ),
            ],
        )
        score, reason = evaluate_loops(conv, None, None, True)
        assert score is not None
        assert score < 1.0
        assert "Exact loop" in reason
        assert "turn t1" in reason

    def test_conversation_level_does_not_join_parent_chains(self) -> None:
        """Parent ids that point at another turn's spans are not a conversation-wide tree."""
        conv = EvaluationData(
            conversation_group_id="c1",
            turns=[
                TurnData(
                    turn_id="t1",
                    query="first",
                    tool_calls=_sequence(_call("root", {}, span_id="s1")),
                ),
                TurnData(
                    turn_id="t2",
                    query="second",
                    tool_calls=_sequence(
                        _call("child", {}, span_id="s2", parent_span_id="s1"),
                        _call("leaf", {}, span_id="s3", parent_span_id="s2"),
                    ),
                ),
            ],
        )
        config = LoopEvalConfig(
            exact_loop_threshold=10,
            soft_loop_threshold=10,
            max_recursive_depth=2,
        )
        score, reason = evaluate_loops(conv, None, None, True, config=config)
        assert score == 1.0
        assert "Excessive recursive depth" not in reason

    def test_conversation_level_missing_all_tool_calls(self) -> None:
        """Conversation with no tool_calls on any turn is an error."""
        conv = EvaluationData(
            conversation_group_id="c1",
            turns=[TurnData(turn_id="t1", query="q", response="r")],
        )
        score, reason = evaluate_loops(conv, None, None, True)
        assert score is None
        assert "No tool_calls provided on turn t1" in reason

    def test_conversation_level_mixed_none_is_error(self) -> None:
        """Missing tool_calls on any turn is an error, not skipped."""
        repeated = _call("search", {"q": "same"})
        conv = EvaluationData(
            conversation_group_id="c1",
            turns=[
                TurnData(
                    turn_id="t1",
                    query="first",
                    tool_calls=_sequence(repeated, repeated),
                ),
                TurnData(turn_id="t2", query="second", tool_calls=None),
                TurnData(
                    turn_id="t3",
                    query="third",
                    tool_calls=_sequence(repeated),
                ),
            ],
        )
        score, reason = evaluate_loops(conv, None, None, True)
        assert score is None
        assert "No tool_calls provided on turn t2" in reason

    def test_conversation_requires_conv_data(self) -> None:
        """Conversation-level evaluation without conv_data is an error."""
        score, reason = evaluate_loops(None, None, None, True)
        assert score is None
        assert "Conversation data is required" in reason


class TestEvaluateLoopsFromTrace:
    """Tests for trace-based loop and depth detection."""

    def test_no_tool_spans_scores_perfect(self) -> None:
        """A trace with only an agent span has no tool loops."""
        trace = Trace(
            trace_id="t1",
            spans=[
                Span(
                    span_id="root",
                    trace_id="t1",
                    name="agent",
                    span_type=SpanType.AGENT,
                )
            ],
        )
        score, reason = evaluate_loops_from_trace(trace)
        assert score == 1.0
        assert "No loops detected" in reason

    def test_exact_loop_on_tool_spans(self) -> None:
        """Consecutive TOOL spans with the same inputs are an exact loop."""
        spans = [
            Span(
                span_id=f"s{i}",
                trace_id="t1",
                name="search",
                span_type=SpanType.TOOL,
                start_time=datetime(2024, 1, 1, 12, 0, i, tzinfo=UTC),
                inputs={"q": "pods"},
            )
            for i in range(3)
        ]
        trace = Trace(trace_id="t1", spans=spans)
        score, reason = evaluate_loops_from_trace(trace)
        assert score < 1.0
        assert "Exact loop" in reason
        assert "search" in reason

    def test_recursive_depth_walks_all_span_types(self) -> None:
        """Depth counts AGENT and TOOL ancestors, not only consecutive tools."""
        trace = Trace(
            trace_id="t1",
            spans=[
                Span(
                    span_id="agent",
                    trace_id="t1",
                    name="agent",
                    span_type=SpanType.AGENT,
                ),
                Span(
                    span_id="chain",
                    trace_id="t1",
                    name="chain",
                    span_type=SpanType.CHAIN,
                    parent_span_id="agent",
                ),
                Span(
                    span_id="tool",
                    trace_id="t1",
                    name="oc_get",
                    span_type=SpanType.TOOL,
                    parent_span_id="chain",
                    inputs={"kind": "pod"},
                ),
            ],
        )
        config = LoopEvalConfig(
            exact_loop_threshold=10,
            soft_loop_threshold=10,
            max_recursive_depth=2,
        )
        score, reason = evaluate_loops_from_trace(trace, config=config)
        assert score < 1.0
        assert "Excessive recursive depth: 3" in reason
        assert "agent -> chain -> oc_get" in reason
        assert "tool call index 0" in reason


class TestLoopEvalConfig:
    """Direct-construction validation for LoopEvalConfig."""

    def test_defaults_are_valid(self) -> None:
        """Default thresholds construct without error."""
        config = LoopEvalConfig()
        assert config.exact_loop_threshold == 3
        assert config.soft_loop_threshold == 3
        assert config.max_recursive_depth == 10

    @pytest.mark.parametrize(
        "field",
        ["exact_loop_threshold", "soft_loop_threshold", "max_recursive_depth"],
    )
    @pytest.mark.parametrize("value", [0, -1, True, 1.5, "3", None])
    def test_invalid_threshold_raises(self, field: str, value: Any) -> None:
        """Zero, negative, or non-int thresholds raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match=field):
            LoopEvalConfig(**{field: value})


class TestConfigFromMetadata:
    """Tests for metadata coercion."""

    def test_defaults_when_empty(self) -> None:
        """Missing metadata uses defaults."""
        config = config_from_metadata(None)
        assert config.exact_loop_threshold == 3
        assert config.soft_loop_threshold == 3
        assert config.max_recursive_depth == 10

    def test_invalid_values_fall_back(self) -> None:
        """Boolean, non-positive, or non-int values keep defaults."""
        config = config_from_metadata(
            {
                "exact_loop_threshold": True,
                "soft_loop_threshold": "nope",
                "max_recursive_depth": 8,
            }
        )
        assert config.exact_loop_threshold == 3
        assert config.soft_loop_threshold == 3
        assert config.max_recursive_depth == 8

    def test_non_positive_metadata_falls_back(self) -> None:
        """Zero or negative metadata values keep defaults instead of raising."""
        config = config_from_metadata(
            {
                "exact_loop_threshold": 0,
                "soft_loop_threshold": -1,
                "max_recursive_depth": 0,
            }
        )
        assert config.exact_loop_threshold == 3
        assert config.soft_loop_threshold == 3
        assert config.max_recursive_depth == 10
