"""Unit tests for the EvaluationSummary model."""

from typing import Any

from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models.data import (
    EvaluationData,
    EvaluationResult,
    TurnData,
)
from lightspeed_evaluation.core.models.summary import (
    EvaluationSummary,
    OverallStats,
    ScoreStatistics,
)


_RESULT_DEFAULTS: dict[str, Any] = {
    "conversation_group_id": "conv1",
    "tag": "eval",
    "turn_id": "turn1",
    "metric_identifier": "ragas:faithfulness",
    "result": "PASS",
    "score": 0.85,
    "threshold": 0.7,
    "reason": "Good",
    "execution_time": 1.0,
    "judge_llm_input_tokens": 100,
    "judge_llm_output_tokens": 50,
}


def _make_result(**overrides: Any) -> EvaluationResult:
    """Create a sample EvaluationResult for testing."""
    kwargs = {**_RESULT_DEFAULTS, **overrides}
    return EvaluationResult(**kwargs)


class TestOverallStatsAndScoreStatistics:
    """Tests for OverallStats and ScoreStatistics models."""

    def test_overall_stats_creation(self) -> None:
        """Test basic creation of OverallStats."""
        stats = OverallStats(
            total=10,
            passed=7,
            failed=2,
            error=1,
            skipped=0,
            pass_rate=70.0,
            fail_rate=20.0,
            error_rate=10.0,
            skipped_rate=0.0,
            total_judge_llm_input_tokens=1000,
            total_judge_llm_output_tokens=500,
            total_judge_llm_tokens=1500,
        )
        assert stats.total == 10
        assert stats.passed == 7
        assert stats.pass_rate == 70.0

    def test_score_statistics_creation(self) -> None:
        """Test basic creation of ScoreStatistics."""
        stats = ScoreStatistics(
            count=5,
            mean=0.8,
            median=0.85,
            std=0.1,
            min_score=0.5,
            max_score=1.0,
        )
        assert stats.count == 5
        assert stats.mean == 0.8
        assert stats.confidence_interval is None

    def test_score_statistics_with_confidence_interval(self) -> None:
        """Test ScoreStatistics with confidence interval."""
        ci: dict[str, float] = {
            "low": 0.7,
            "mean": 0.8,
            "high": 0.9,
            "confidence_level": 95,
        }
        stats = ScoreStatistics(
            count=5,
            mean=0.8,
            median=0.85,
            std=0.1,
            min_score=0.5,
            max_score=1.0,
            confidence_interval=ci,
        )
        assert stats.confidence_interval is not None
        assert stats.model_dump()["confidence_interval"]["low"] == 0.7


class TestEvaluationSummaryFromResults:
    """Tests for EvaluationSummary.from_results() classmethod."""

    def test_typical_results(self) -> None:
        """Test from_results with a mix of PASS, FAIL, ERROR results."""
        results = [
            _make_result(
                result="PASS", score=0.9, metric_identifier="ragas:faithfulness"
            ),
            _make_result(
                result="FAIL",
                score=0.4,
                metric_identifier="ragas:answer_relevancy",
                conversation_group_id="conv2",
                turn_id="turn2",
            ),
            _make_result(
                result="ERROR",
                score=None,
                metric_identifier="ragas:faithfulness",
                conversation_group_id="conv2",
                turn_id="turn3",
            ),
        ]

        summary = EvaluationSummary.from_results(results)

        assert summary.overall.total == 3
        assert summary.overall.passed == 1
        assert summary.overall.failed == 1
        assert summary.overall.error == 1
        assert summary.overall.skipped == 0

        # Check by_metric
        assert "ragas:faithfulness" in summary.by_metric
        assert "ragas:answer_relevancy" in summary.by_metric

        # Check by_conversation
        assert "conv1" in summary.by_conversation
        assert "conv2" in summary.by_conversation

        # Check timestamp is set
        assert summary.timestamp is not None

    def test_empty_results(self) -> None:
        """Test from_results with empty results list."""
        summary = EvaluationSummary.from_results([])

        assert summary.overall.total == 0
        assert summary.overall.passed == 0
        assert summary.overall.pass_rate == 0.0
        assert not summary.by_metric
        assert not summary.by_conversation
        assert not summary.by_tag

    def test_all_error_results(self) -> None:
        """Test from_results with all ERROR results."""
        results = [
            _make_result(result="ERROR", score=None, metric_identifier="m:1"),
            _make_result(result="ERROR", score=None, metric_identifier="m:2"),
        ]

        summary = EvaluationSummary.from_results(results)

        assert summary.overall.total == 2
        assert summary.overall.error == 2
        assert summary.overall.error_rate == 100.0
        assert summary.overall.passed == 0

    def test_with_evaluation_data(self) -> None:
        """Test from_results with evaluation_data for api tokens and streaming."""
        results = [_make_result()]

        eval_data = [
            EvaluationData(
                conversation_group_id="conv1",
                turns=[
                    TurnData(
                        turn_id="turn1",
                        query="What is OCP?",
                        api_input_tokens=500,
                        api_output_tokens=200,
                        time_to_first_token=0.5,
                        streaming_duration=2.0,
                        tokens_per_second=100.0,
                    ),
                ],
            )
        ]

        summary = EvaluationSummary.from_results(results, evaluation_data=eval_data)

        assert summary.api_tokens is not None
        assert summary.api_tokens.total_api_input_tokens == 500
        assert summary.api_tokens.total_api_output_tokens == 200
        assert summary.api_tokens.total_api_tokens == 700

        assert summary.streaming is not None
        assert summary.streaming.time_to_first_token is not None
        assert summary.streaming.time_to_first_token.mean == 0.5

    def test_without_confidence_intervals_by_default(self) -> None:
        """Test that confidence intervals are not computed by default."""
        results = [
            _make_result(score=0.8, metric_identifier="m:1"),
            _make_result(score=0.9, metric_identifier="m:1", turn_id="t2"),
            _make_result(score=0.7, metric_identifier="m:1", turn_id="t3"),
        ]

        summary = EvaluationSummary.from_results(results)

        metric_stats = summary.by_metric["m:1"]
        assert metric_stats.score_statistics is not None
        assert metric_stats.score_statistics.confidence_interval is None

    def test_with_confidence_intervals(self, mocker: MockerFixture) -> None:
        """Test from_results with compute_confidence_intervals=True."""
        results = [
            _make_result(score=0.8, metric_identifier="m:1"),
            _make_result(score=0.9, metric_identifier="m:1", turn_id="t2"),
            _make_result(score=0.7, metric_identifier="m:1", turn_id="t3"),
        ]

        # Mock bootstrap_intervals to avoid the expensive 10k iteration computation
        mocker.patch(
            "lightspeed_evaluation.core.models.summary.bootstrap_intervals",
            return_value=(0.72, 0.8, 0.88),
        )

        summary = EvaluationSummary.from_results(
            results, compute_confidence_intervals=True
        )

        metric_stats = summary.by_metric["m:1"]
        assert metric_stats.score_statistics is not None
        ci = metric_stats.score_statistics.confidence_interval
        assert ci is not None
        assert ci["low"] == 0.72

    def test_by_tag_stats(self) -> None:
        """Test that by_tag statistics are computed correctly."""
        results = [
            _make_result(tag="qa", result="PASS", score=0.9, metric_identifier="m:1"),
            _make_result(
                tag="qa",
                result="FAIL",
                score=0.3,
                metric_identifier="m:1",
                turn_id="t2",
            ),
            _make_result(tag="perf", result="PASS", score=0.8, metric_identifier="m:2"),
        ]

        summary = EvaluationSummary.from_results(results)

        assert "qa" in summary.by_tag
        assert "perf" in summary.by_tag
        assert summary.by_tag["qa"].passed == 1
        assert summary.by_tag["qa"].failed == 1
        assert summary.by_tag["perf"].passed == 1

    def test_results_list_stored(self) -> None:
        """Test that results list is stored in summary."""
        results = [_make_result()]
        summary = EvaluationSummary.from_results(results)
        assert len(summary.results) == 1
        assert summary.results[0].result == "PASS"


class TestEvaluationSummaryModelDumpAndValidation:
    """Tests for EvaluationSummary.model_dump() and Pydantic validation."""

    def test_model_dump_structure(self) -> None:
        """Test that model_dump produces expected structure."""
        results = [
            _make_result(result="PASS", score=0.9),
            _make_result(
                result="FAIL", score=0.4, turn_id="t2", metric_identifier="m:2"
            ),
        ]

        summary = EvaluationSummary.from_results(results)
        dumped = summary.model_dump()

        assert "timestamp" in dumped
        assert "overall" in dumped
        assert "by_metric" in dumped
        assert "by_conversation" in dumped
        assert "by_tag" in dumped
        assert "results" in dumped

        # Check overall structure
        overall = dumped["overall"]
        assert "total" in overall
        assert "passed" in overall
        assert "failed" in overall
        assert "pass_rate" in overall

    def test_model_dump_excludes_none_optional(self) -> None:
        """Test that None optional fields are handled properly in dump."""
        results = [_make_result()]
        summary = EvaluationSummary.from_results(results)
        dumped = summary.model_dump()

        # api_tokens and streaming should be None when not provided
        assert dumped["api_tokens"] is None
        assert dumped["streaming"] is None

    def test_pydantic_validation(self) -> None:
        """Test that the model validates correctly via Pydantic."""
        summary = EvaluationSummary(
            timestamp="2025-01-01T00:00:00",
            results=[],
            overall=OverallStats(
                total=0,
                passed=0,
                failed=0,
                error=0,
                skipped=0,
                pass_rate=0.0,
                fail_rate=0.0,
                error_rate=0.0,
                skipped_rate=0.0,
                total_judge_llm_input_tokens=0,
                total_judge_llm_output_tokens=0,
                total_judge_llm_tokens=0,
            ),
            by_metric={},
            by_conversation={},
            by_tag={},
        )
        assert summary.timestamp == "2025-01-01T00:00:00"
        assert summary.overall.total == 0
