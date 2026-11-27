"""Unit tests for the native-scale fields a metric result can carry.

``score`` stays normalised to [0, 1] for statistics and reporting; these
fields keep the metric's own value, scale and pass rule alongside it, and
record the agent the result belongs to.
"""

from lightspeed_evaluation.core.models.data import (
    EvaluationResult,
    MetricResult,
    ScoredMetric,
)


class TestMetricResultNativeScale:
    """Tests for MetricResult.raw_score / score_min / score_max."""

    def test_native_scale_fields_default_to_none(self) -> None:
        """Metrics on the usual [0, 1] scale leave the native-scale fields unset."""
        result = MetricResult(result="PASS", score=0.8, threshold=0.7)

        assert result.raw_score is None
        assert result.score_min is None
        assert result.score_max is None

    def test_raw_score_keeps_value_outside_zero_one(self) -> None:
        """A signed native score survives alongside the normalised score."""
        result = MetricResult(
            result="FAIL",
            score=0.4,
            threshold=1.0,
            raw_score=-0.6,
            score_min=-1.0,
            score_max=1.0,
            reason="1 - |-0.6| = 0.4",
        )

        assert result.raw_score == -0.6
        assert (result.score_min, result.score_max) == (-1.0, 1.0)
        assert result.score == 0.4

    def test_pass_condition_is_the_metric_s_own_rule(self) -> None:
        """Distinct from 'threshold', which applies to the reported score."""
        result = MetricResult(
            result="FAIL",
            score=0.4,
            threshold=1.0,
            raw_score=-0.6,
            pass_condition="between(0.0, 0.0)",
        )

        assert result.pass_condition == "between(0.0, 0.0)"
        assert result.threshold == 1.0

    def test_evaluation_result_inherits_native_scale_fields(self) -> None:
        """EvaluationResult carries the fields through to output."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="uaem:granularity_fit",
            result="FAIL",
            score=0.0,
            raw_score=1.0,
            score_min=-1.0,
            score_max=1.0,
        )

        assert result.raw_score == 1.0


class TestScoredMetric:
    """Tests for the handler result that carries a native scale."""

    def test_two_tuple_is_normalized(self) -> None:
        """A handler on the old contract yields a result with no native fields."""
        scored = ScoredMetric.from_handler((0.85, "Good"))

        assert scored.score == 0.85
        assert scored.reason == "Good"
        assert scored.raw_score is None

    def test_a_scored_metric_passes_through_unchanged(self) -> None:
        """A handler that already returns the typed result is left alone."""
        original = ScoredMetric(score=0.0, raw_score=-1.0, score_min=-1.0)

        assert ScoredMetric.from_handler(original) is original


class TestEvaluationResultAgent:
    """Tests for the agent recorded on each result."""

    def test_agent_defaults_to_none(self) -> None:
        """An offline run names no agent for the responses it scores."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.9,
        )

        assert result.agent is None

    def test_agent_is_recorded_for_grouping(self) -> None:
        """Finalization groups by the system scored, not by the scorer."""
        result = EvaluationResult(
            conversation_group_id="conv1",
            agent="agent_a",
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.9,
        )

        assert result.agent == "agent_a"
