"""Unit tests for quality report model."""

from pytest import LogCaptureFixture

from lightspeed_evaluation.core.models.quality import QualityReport
from lightspeed_evaluation.core.models.statistics import (
    AgentTokenStats,
    MetricStats,
    NumericStats,
)


class TestQualityReport:
    """Tests for QualityReport model and create_report() method."""

    def test_quality_report_creation_happy_path(
        self,
        quality_by_metric: dict[str, MetricStats],
        api_latency_summary: NumericStats,
    ) -> None:
        """Test QualityReport creation with valid metrics."""
        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric, api_latency_summary, None, quality_score_metrics
        )

        # Assertions
        assert report is not None

        # Check that quality metrics are correctly separated
        assert len(report.quality_metrics) == 2
        assert "ragas:faithfulness" in report.quality_metrics
        assert "ragas:answer_relevancy" in report.quality_metrics

        # Check that extra metrics contain the non-quality metrics
        assert len(report.extra_metrics) == 1
        assert "custom:context_recall" in report.extra_metrics

        # Each metric has 10 samples, so weights should be 0.5 each
        assert report.quality_metrics["ragas:faithfulness"].weight == 0.5
        assert report.quality_metrics["ragas:answer_relevancy"].weight == 0.5
        # Verify weights are calculated correctly (should sum to 1.0)
        total_weight = sum(metric.weight for metric in report.quality_metrics.values())
        assert total_weight == 1.0

        # Verify aggregated quality score is weighted average
        # Expected: (0.85 * 0.5) + (0.90 * 0.5) = 0.875
        expected_score = (0.85 * 0.5) + (0.90 * 0.5)
        assert report.quality_score == expected_score

        # Verify quality metrics contain correct mean scores
        assert report.quality_metrics["ragas:faithfulness"].statistics.mean == 0.85
        assert report.quality_metrics["ragas:answer_relevancy"].statistics.mean == 0.90

        # Verify extra metrics contain correct mean scores
        assert report.extra_metrics["custom:context_recall"].mean == 0.75

        # Verify agent latency is set correctly
        assert report.agent_latency_stats is not None
        assert report.agent_latency_stats.count == 10
        assert report.agent_latency_stats.mean == 1.5

        # Verify no warnings for valid configuration
        assert len(report.warnings) == 0

    def test_quality_report_creation_missing_metric(
        self,
        quality_by_metric: dict[str, MetricStats],
        api_latency_summary: NumericStats,
    ) -> None:
        """Test QualityReport excludes missing metrics and generates warning."""
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_correctness"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric, api_latency_summary, None, quality_score_metrics
        )

        # Assertions
        assert report is not None

        # Check that quality metrics are correctly separated
        assert len(report.quality_metrics) == 1
        assert "ragas:faithfulness" in report.quality_metrics
        assert report.quality_metrics["ragas:faithfulness"].weight == 1.0
        assert (
            report.quality_metrics["ragas:faithfulness"].statistics.mean
            == report.quality_score
        )

        # Verify warning about missing metric
        assert len(report.warnings) == 1
        assert any(
            "ragas:answer_correctness" in warning and "excluded" in warning
            for warning in report.warnings
        )

        # Check that extra metrics contain the non-quality metrics
        assert len(report.extra_metrics) == 2
        assert "custom:context_recall" in report.extra_metrics
        assert "ragas:answer_relevancy" in report.extra_metrics

    def test_quality_report_total_samples_zero(
        self,
        quality_by_metric_zero: dict[str, MetricStats],
        api_latency_summary: NumericStats,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test QualityReport returns None when all quality metrics have zero samples."""
        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric_zero, api_latency_summary, None, quality_score_metrics
        )

        # Assertions
        assert report is None
        assert "Quality score computation failed" in caplog.text

    def test_quality_report_sample_size_zero(
        self,
        quality_by_metric_zero: dict[str, MetricStats],
        api_latency_summary: NumericStats,
    ) -> None:
        """Test QualityReport excludes metrics with zero samples and generates warning."""
        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "custom:context_recall"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric_zero, api_latency_summary, None, quality_score_metrics
        )

        # Assertions
        assert report is not None
        assert any(
            "ragas:faithfulness" in warning and "excluded" in warning
            for warning in report.warnings
        )

    def test_quality_report_none_score_statistics(
        self,
        quality_by_metric_with_none: dict[str, MetricStats],
        api_latency_summary: NumericStats,
        caplog: LogCaptureFixture,
    ) -> None:
        """Test QualityReport excludes metrics with None score_statistics and logs warning."""
        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric_with_none,
            api_latency_summary,
            None,
            quality_score_metrics,
        )

        # Assertions
        assert report is not None

        # Check that the metric with None score_statistics was excluded
        assert len(report.quality_metrics) == 1
        assert "ragas:answer_relevancy" in report.quality_metrics
        assert "ragas:faithfulness" not in report.quality_metrics

        # Verify weight is 1.0 since only one metric remains
        assert report.quality_metrics["ragas:answer_relevancy"].weight == 1.0

        # Verify aggregated quality score equals the single metric's mean
        assert report.quality_score == 0.90

        # Verify warning was generated and logged
        assert len(report.warnings) == 1
        assert any(
            "ragas:faithfulness" in warning
            and "excluded" in warning
            and "Missing score statistics data" in warning
            for warning in report.warnings
        )

        # Verify warning was logged
        assert "ragas:faithfulness" in caplog.text
        assert "Missing score statistics data" in caplog.text

    def test_quality_report_creation_no_api_latency(
        self,
        quality_by_metric: dict[str, MetricStats],
    ) -> None:
        """Test QualityReport handles None API latency (api_enabled=False)."""
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]
        api_latency_summary = None

        # Create the QualityReport with None agent_latency_stats
        report = QualityReport.create_report(
            quality_by_metric,
            api_latency_summary,  # API disabled scenario
            None,  # No agent token stats
            quality_score_metrics,
        )

        # Assertions
        assert report is not None
        assert report.agent_latency_stats is None  # Should gracefully handle None
        assert report.quality_score > 0  # Quality score still computed
        assert len(report.quality_metrics) == 2

    def test_quality_report_with_agent_token_stats(
        self,
        quality_by_metric: dict[str, MetricStats],
        api_latency_summary: NumericStats,
        agent_token_stats: AgentTokenStats,
    ) -> None:
        """Test QualityReport includes agent token statistics with percentiles."""
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport with agent token stats
        report = QualityReport.create_report(
            quality_by_metric,
            api_latency_summary,
            agent_token_stats,
            quality_score_metrics,
        )

        # Assertions
        assert report is not None
        assert report.agent_token_stats is not None

        # Verify input token statistics
        assert report.agent_token_stats.input is not None
        assert report.agent_token_stats.input.count == 10
        assert report.agent_token_stats.input.mean == 450.5
        assert report.agent_token_stats.input.median == 425.0
        assert report.agent_token_stats.input.p95 == 520.0
        assert report.agent_token_stats.input.p99 == 545.0

        # Verify output token statistics
        assert report.agent_token_stats.output is not None
        assert report.agent_token_stats.output.count == 10
        assert report.agent_token_stats.output.mean == 180.3
        assert report.agent_token_stats.output.median == 175.0
        assert report.agent_token_stats.output.p95 == 210.0
        assert report.agent_token_stats.output.p99 == 218.0

    def test_quality_report_with_no_agent_token_stats(
        self,
        quality_by_metric: dict[str, MetricStats],
        api_latency_summary: NumericStats,
    ) -> None:
        """Test QualityReport handles None agent token stats gracefully."""
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport without agent token stats
        report = QualityReport.create_report(
            quality_by_metric,
            api_latency_summary,
            None,  # No agent token stats
            quality_score_metrics,
        )

        # Assertions
        assert report is not None
        assert report.agent_token_stats is None
        assert report.quality_score > 0  # Quality score still computed
