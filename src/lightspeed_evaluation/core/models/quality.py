"""Quality score models for aggregated quality assessment.

Provides Pydantic models for computing and reporting an aggregated quality score
from selected metrics using weighted averaging based on sample sizes.
"""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from lightspeed_evaluation.core.models.statistics import (
    MetricStats,
    NumericStats,
    ScoreStatistics,
    AgentTokenStats,
)


logger = logging.getLogger(__name__)


class QualityMetricResult(BaseModel):
    """Quality metric result using composition to add weight to score statistics."""

    statistics: ScoreStatistics = Field(
        description="Score statistics for this quality metric"
    )
    weight: float = Field(
        default=0.0,
        description="Weight proportion (sample_size / total_samples) used in weighted average",
    )


class QualityReport(BaseModel):
    """Aggregated quality score from selected metrics."""

    quality_score: float = Field(
        default=0.0, description="Weighted average of quality score metrics"
    )
    quality_metrics: dict[str, QualityMetricResult] = Field(
        default_factory=dict,
        description="Individual metrics used in quality score calculation",
    )
    extra_metrics: dict[str, ScoreStatistics] = Field(
        default_factory=dict,
        description="Other evaluated metrics calculated, not used for quality score calculation",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about quality metrics configuration or usage",
    )
    agent_latency_stats: Optional[NumericStats] = Field(
        default=None, description="Agent latency statistics"
    )
    agent_token_stats: Optional[AgentTokenStats] = Field(
        default=None, description="Agent token usage statistics"
    )

    @staticmethod
    def create_report(
        by_metric: dict[str, MetricStats],
        agent_latency_stats: Optional[NumericStats],
        agent_token_stats: Optional[AgentTokenStats],
        quality_score_metrics: list[str],
    ) -> Optional["QualityReport"]:
        """Creates a quality report with aggregated quality score from selected metrics.

        Separates metrics into quality metrics (used for quality score calculation) and
        extra metrics (evaluated but not included in quality score).

        Args:
            by_metric: Dictionary mapping metric identifiers to their computed statistics.
            agent_latency_stats: Agent API latency statistics (p50, p95, p99).
            agent_token_stats: Agent token usage statistics with percentiles.
            quality_score_metrics: Metric identifiers to include in quality score calculation.
                All specified metrics must exist in by_metric.

        Returns:
            QualityReport with aggregated quality score and separated quality/extra metrics,
            or None if all quality score metrics have zero samples.

        Raises:
            ValueError: If any quality_score_metrics are not found in by_metric.
        """
        warnings: list[str] = []

        # Validate all quality score metrics exist in computed metrics (by_metric)
        missing_metrics = [m for m in quality_score_metrics if m not in by_metric]
        if missing_metrics:
            warning_msg = (
                "WARNING: "
                f"Quality score metrics {missing_metrics} were excluded from "
                "quality score computation. "
                f"Reason: Not found in evaluation results."
            )
            warnings.append(warning_msg)
            logger.warning(warning_msg)

            quality_score_metrics = list(
                set(quality_score_metrics) - set(missing_metrics)
            )

        # Calculate total samples from quality score metrics only
        total_samples = 0
        for metric_id in quality_score_metrics:
            score_stats = by_metric[metric_id].score_statistics
            if score_stats is not None:
                total_samples += score_stats.count
        if total_samples == 0:
            logger.warning(
                "CRITICAL: Quality score computation failed. "
                "All configured quality metrics have zero evaluation results."
            )
            return None

        quality_metrics: dict[str, QualityMetricResult] = {}
        extra_metrics: dict[str, ScoreStatistics] = {}

        # Separate quality metrics from extra metrics
        for metric_id in by_metric:
            if metric_id in quality_score_metrics:
                score_stats = by_metric[metric_id].score_statistics

                # Skip if score_statistics is None
                if score_stats is None:
                    warning_msg = (
                        f"WARNING: Quality score metric '{metric_id}' "
                        "was excluded from quality score computation. "
                        "Reason: Missing score statistics data."
                    )
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    continue

                sample_size = score_stats.count

                # Skip metrics with zero samples
                if sample_size == 0:
                    warning_msg = (
                        f"WARNING: Quality score metric '{metric_id}' "
                        "was excluded from quality score computation. "
                        "Reason: Zero evaluation results for this metric."
                    )
                    warnings.append(warning_msg)
                    logger.warning(warning_msg)
                    continue

                weight = sample_size / total_samples

                quality_metrics[metric_id] = QualityMetricResult(
                    statistics=score_stats,
                    weight=weight,
                )
            else:
                stats = by_metric[metric_id].score_statistics
                if stats is not None:
                    extra_metrics[metric_id] = stats

        return QualityReport(
            quality_score=QualityReport._calculate_quality_score(quality_metrics),
            quality_metrics=quality_metrics,
            extra_metrics=extra_metrics,
            warnings=warnings,
            agent_latency_stats=agent_latency_stats,
            agent_token_stats=agent_token_stats,
        )

    @staticmethod
    def _calculate_quality_score(
        quality_metrics: dict[str, QualityMetricResult],
    ) -> float:
        """Calculate weighted average quality score from quality metrics.

        Computes a weighted average where each metric's weight is proportional to its
        sample size relative to the total samples across all quality metrics.

        Args:
            quality_metrics: Dictionary of quality metric results with statistics and
                weights. Each metric contains statistics with a mean score and a weight
                (sample_size / total_samples).

        Returns:
            Weighted average quality score computed as sum of (mean * weight) for all metrics.
        """
        weighted_sum = 0.0
        for metric in quality_metrics.values():
            weighted_sum += metric.statistics.mean * metric.weight
        return weighted_sum
