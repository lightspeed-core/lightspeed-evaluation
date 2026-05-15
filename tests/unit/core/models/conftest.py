"""Fixtures for models tests."""

import pytest

from lightspeed_evaluation.core.models import MetricStats, ScoreStatistics


@pytest.fixture
def quality_by_metric() -> dict[str, MetricStats]:
    """Sample MetricStats with non-zero counts for quality report tests."""
    return {
        "ragas:faithfulness": MetricStats(
            score_statistics=ScoreStatistics(
                count=10,
                mean=0.85,
            ),
        ),
        "ragas:answer_relevancy": MetricStats(
            score_statistics=ScoreStatistics(
                count=10,
                mean=0.90,
            ),
        ),
        "custom:context_recall": MetricStats(
            score_statistics=ScoreStatistics(
                count=10,
                mean=0.75,
            ),
        ),
    }


@pytest.fixture
def quality_by_metric_zero() -> dict[str, MetricStats]:
    """Sample MetricStats with some zero counts for quality report tests."""
    return {
        "ragas:faithfulness": MetricStats(
            score_statistics=ScoreStatistics(
                count=0,
                mean=0.0,
            ),
        ),
        "ragas:answer_relevancy": MetricStats(
            score_statistics=ScoreStatistics(
                count=0,
                mean=0.0,
            ),
        ),
        "custom:context_recall": MetricStats(
            score_statistics=ScoreStatistics(
                count=10,
                mean=0.75,
            ),
        ),
    }


@pytest.fixture
def quality_by_metric_with_none() -> dict[str, MetricStats]:
    """Sample MetricStats with None score_statistics for quality report tests."""
    return {
        "ragas:faithfulness": MetricStats(
            score_statistics=None,
        ),
        "ragas:answer_relevancy": MetricStats(
            score_statistics=ScoreStatistics(
                count=10,
                mean=0.90,
            ),
        ),
        "custom:context_recall": MetricStats(
            score_statistics=ScoreStatistics(
                count=10,
                mean=0.75,
            ),
        ),
    }
