"""Fixtures for models tests."""

import pytest

from lightspeed_evaluation.core.models.statistics import (
    AgentTokenStats,
    MetricStats,
    NumericStats,
    ScoreStatistics,
)


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


@pytest.fixture
def api_latency_summary() -> NumericStats:
    """Sample API latency statistics for quality report tests."""
    return NumericStats(
        count=10,
        mean=1.5,
        median=1.3,
        std=0.5,
        min_value=0.8,
        max_value=2.5,
        p95=2.2,
        p99=2.4,
    )


@pytest.fixture
def agent_token_stats() -> AgentTokenStats:
    """Sample agent token statistics for quality report tests."""
    return AgentTokenStats(
        input=NumericStats(
            count=10,
            mean=450.5,
            median=425.0,
            std=50.2,
            min_value=350.0,
            max_value=550.0,
            p95=520.0,
            p99=545.0,
        ),
        output=NumericStats(
            count=10,
            mean=180.3,
            median=175.0,
            std=25.1,
            min_value=140.0,
            max_value=220.0,
            p95=210.0,
            p99=218.0,
        ),
    )
