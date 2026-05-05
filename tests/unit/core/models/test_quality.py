"""Unit tests for quality report model."""

from pytest import LogCaptureFixture

from lightspeed_evaluation.core.models.quality import QualityReport
from lightspeed_evaluation.core.models.summary import MetricStats


class TestQualityReport:
    """Tests for QualityReport model and create_report() method."""

    def test_quality_report_creation_happy_path(
        self,
        quality_by_metric: dict[str, MetricStats],
    ) -> None:
        """Test QualityReport creation with valid metrics."""

        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport
        report = QualityReport.create_report(quality_by_metric, quality_score_metrics)

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

        # Verify no warnings for valid configuration
        assert len(report.warnings) == 0

    def test_quality_report_creation_missing_metric(
        self, quality_by_metric: dict[str, MetricStats]
    ) -> None:
        """Test QualityReport excludes missing metrics and generates warning."""
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_correctness"]

        # Create the QualityReport
        report = QualityReport.create_report(quality_by_metric, quality_score_metrics)

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
        self, quality_by_metric_zero: dict[str, MetricStats], caplog: LogCaptureFixture
    ) -> None:
        """Test QualityReport returns None when all quality metrics have zero samples."""
        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric_zero, quality_score_metrics
        )

        # Assertions
        assert report is None
        assert "Quality score computation failed" in caplog.text

    def test_quality_report_sample_size_zero(
        self, quality_by_metric_zero: dict[str, MetricStats]
    ) -> None:
        """Test QualityReport excludes metrics with zero samples and generates warning."""
        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "custom:context_recall"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric_zero, quality_score_metrics
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
        caplog: LogCaptureFixture,
    ) -> None:
        """Test QualityReport excludes metrics with None score_statistics and logs warning."""
        # Define quality score metrics (subset of all metrics)
        quality_score_metrics = ["ragas:faithfulness", "ragas:answer_relevancy"]

        # Create the QualityReport
        report = QualityReport.create_report(
            quality_by_metric_with_none, quality_score_metrics
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
