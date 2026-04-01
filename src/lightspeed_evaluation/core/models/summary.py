"""Evaluation summary models for structured results.

Provides Pydantic models that wrap evaluation results with computed statistics,
decoupling result computation from file persistence.
"""

import statistics as stats_module
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from lightspeed_evaluation.core.models.data import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.output.statistics import (
    bootstrap_intervals,
    calculate_api_token_usage,
    calculate_streaming_stats,
)


class NumericStats(BaseModel):
    """Numeric statistics for a set of values (e.g., TTFT, duration)."""

    count: int = Field(default=0, description="Number of values")
    mean: Optional[float] = Field(default=None, description="Mean value")
    median: Optional[float] = Field(default=None, description="Median value")
    std: Optional[float] = Field(default=None, description="Standard deviation")
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")


class ScoreStatistics(BaseModel):
    """Score statistics for a metric or group."""

    count: int = Field(default=0, description="Number of scored results")
    mean: float = Field(default=0.0, description="Mean score")
    median: float = Field(default=0.0, description="Median score")
    std: float = Field(default=0.0, description="Standard deviation")
    min_score: float = Field(default=0.0, description="Minimum score")
    max_score: float = Field(default=0.0, description="Maximum score")
    confidence_interval: Optional[dict[str, float]] = Field(
        default=None,
        description="Bootstrap confidence interval with low, mean, high, confidence_level",
    )


class OverallStats(BaseModel):
    """Overall pass/fail/error/skipped statistics."""

    total: int = Field(default=0, description="Total number of evaluations")
    passed: int = Field(default=0, description="Number of passed evaluations")
    failed: int = Field(default=0, description="Number of failed evaluations")
    error: int = Field(default=0, description="Number of error evaluations")
    skipped: int = Field(default=0, description="Number of skipped evaluations")
    pass_rate: float = Field(default=0.0, description="Pass rate percentage")
    fail_rate: float = Field(default=0.0, description="Fail rate percentage")
    error_rate: float = Field(default=0.0, description="Error rate percentage")
    skipped_rate: float = Field(default=0.0, description="Skipped rate percentage")
    total_judge_llm_input_tokens: int = Field(
        default=0, description="Total judge LLM input tokens"
    )
    total_judge_llm_output_tokens: int = Field(
        default=0, description="Total judge LLM output tokens"
    )
    total_judge_llm_tokens: int = Field(default=0, description="Total judge LLM tokens")


class MetricStats(OverallStats):
    """Statistics for a specific metric, extending OverallStats with score statistics."""

    score_statistics: Optional[ScoreStatistics] = Field(
        default=None, description="Score statistics for this metric"
    )


class ConversationStats(OverallStats):
    """Statistics for a specific conversation group."""


class TagStats(OverallStats):
    """Statistics for a specific tag, extending OverallStats with score statistics."""

    score_statistics: Optional[ScoreStatistics] = Field(
        default=None, description="Score statistics for this tag"
    )


class StreamingStats(BaseModel):
    """Streaming performance statistics."""

    time_to_first_token: Optional[NumericStats] = Field(
        default=None, description="Time to first token statistics"
    )
    streaming_duration: Optional[NumericStats] = Field(
        default=None, description="Streaming duration statistics"
    )
    tokens_per_second: Optional[NumericStats] = Field(
        default=None, description="Tokens per second statistics"
    )


class ApiTokenUsage(BaseModel):
    """API token usage totals."""

    total_api_input_tokens: int = Field(default=0, description="Total API input tokens")
    total_api_output_tokens: int = Field(
        default=0, description="Total API output tokens"
    )
    total_api_tokens: int = Field(default=0, description="Total API tokens")


class EvaluationSummary(BaseModel):
    """Structured evaluation summary wrapping results with computed statistics.

    This model decouples result computation from file persistence, enabling
    programmatic access to evaluation statistics without requiring file I/O.
    """

    timestamp: str = Field(
        description="ISO format timestamp of when summary was created"
    )
    results: list[EvaluationResult] = Field(
        default_factory=list, description="Raw evaluation results"
    )
    overall: OverallStats = Field(description="Overall statistics")
    by_metric: dict[str, MetricStats] = Field(
        default_factory=dict, description="Statistics per metric"
    )
    by_conversation: dict[str, ConversationStats] = Field(
        default_factory=dict, description="Statistics per conversation"
    )
    by_tag: dict[str, TagStats] = Field(
        default_factory=dict, description="Statistics per tag"
    )
    api_tokens: Optional[ApiTokenUsage] = Field(
        default=None, description="API token usage (when evaluation data provided)"
    )
    streaming: Optional[StreamingStats] = Field(
        default=None, description="Streaming performance stats (when available)"
    )

    @classmethod
    def from_results(
        cls,
        results: list[EvaluationResult],
        evaluation_data: Optional[list[EvaluationData]] = None,
        compute_confidence_intervals: bool = False,
    ) -> "EvaluationSummary":
        """Create an EvaluationSummary from a list of results.

        Computes overall, per-metric, per-conversation, and per-tag statistics.
        Bootstrap confidence intervals are only computed when explicitly requested.

        Args:
            results: List of evaluation results to summarize.
            evaluation_data: Optional evaluation data for API token and streaming stats.
            compute_confidence_intervals: Whether to compute bootstrap confidence
                intervals (expensive: 10,000 iterations per metric). Default False.

        Returns:
            A fully populated EvaluationSummary instance.
        """
        timestamp = datetime.now().isoformat()

        # Compute overall stats
        overall = _compute_overall_stats(results)

        # Compute per-metric, per-conversation, per-tag stats
        by_metric = _compute_metric_stats(results, compute_confidence_intervals)
        by_conversation = _compute_conversation_stats(results)
        by_tag = _compute_tag_stats(results, compute_confidence_intervals)

        # Compute API token usage and streaming stats if evaluation data provided
        api_tokens = None
        streaming = None
        if evaluation_data:
            api_tokens = _compute_api_token_usage(evaluation_data)
            streaming = _compute_streaming_stats(evaluation_data)

        return cls(
            timestamp=timestamp,
            results=results,
            overall=overall,
            by_metric=by_metric,
            by_conversation=by_conversation,
            by_tag=by_tag,
            api_tokens=api_tokens,
            streaming=streaming,
        )


def _compute_overall_stats(results: list[EvaluationResult]) -> OverallStats:
    """Compute overall statistics from results."""
    if not results:
        return OverallStats()

    total = len(results)
    passed = sum(1 for r in results if r.result == "PASS")
    failed = sum(1 for r in results if r.result == "FAIL")
    error = sum(1 for r in results if r.result == "ERROR")
    skipped = sum(1 for r in results if r.result == "SKIPPED")

    total_judge_input = sum(r.judge_llm_input_tokens for r in results)
    total_judge_output = sum(r.judge_llm_output_tokens for r in results)

    return OverallStats(
        total=total,
        passed=passed,
        failed=failed,
        error=error,
        skipped=skipped,
        pass_rate=(passed / total) * 100 if total > 0 else 0.0,
        fail_rate=(failed / total) * 100 if total > 0 else 0.0,
        error_rate=(error / total) * 100 if total > 0 else 0.0,
        skipped_rate=(skipped / total) * 100 if total > 0 else 0.0,
        total_judge_llm_input_tokens=total_judge_input,
        total_judge_llm_output_tokens=total_judge_output,
        total_judge_llm_tokens=total_judge_input + total_judge_output,
    )


def _compute_group_counts(
    results: list[EvaluationResult],
) -> tuple[int, int, int, int]:
    """Compute pass/fail/error/skipped counts for a group of results."""
    passed = sum(1 for r in results if r.result == "PASS")
    failed = sum(1 for r in results if r.result == "FAIL")
    error = sum(1 for r in results if r.result == "ERROR")
    skipped = sum(1 for r in results if r.result == "SKIPPED")
    return passed, failed, error, skipped


def _compute_rates(
    total: int, passed: int, failed: int, error: int, skipped: int
) -> tuple[float, float, float, float]:
    """Compute rate percentages from counts."""
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        (passed / total) * 100,
        (failed / total) * 100,
        (error / total) * 100,
        (skipped / total) * 100,
    )


def _compute_score_statistics(
    scores: list[float],
    compute_ci: bool = False,
) -> ScoreStatistics:
    """Compute score statistics from a list of scores.

    Args:
        scores: List of numeric scores.
        compute_ci: Whether to compute bootstrap confidence intervals.

    Returns:
        ScoreStatistics instance.
    """
    if not scores:
        return ScoreStatistics()

    count = len(scores)
    mean = stats_module.mean(scores)
    median = stats_module.median(scores)
    std = stats_module.stdev(scores) if count > 1 else 0.0
    min_score = min(scores)
    max_score = max(scores)

    confidence_interval = None
    if compute_ci and count > 1:
        confidence_interval = _try_bootstrap(scores)

    return ScoreStatistics(
        count=count,
        mean=mean,
        median=median,
        std=std,
        min_score=min_score,
        max_score=max_score,
        confidence_interval=confidence_interval,
    )


def _build_group_stats_dict(
    group_results: list[EvaluationResult],
) -> dict[str, Any]:
    """Build a base statistics dictionary for a group of results.

    Args:
        group_results: Results belonging to a single group.

    Returns:
        Dictionary with counts, rates, and token totals.
    """
    total = len(group_results)
    passed, failed, error, skipped = _compute_group_counts(group_results)
    pass_rate, fail_rate, error_rate, skipped_rate = _compute_rates(
        total, passed, failed, error, skipped
    )
    judge_input = sum(r.judge_llm_input_tokens for r in group_results)
    judge_output = sum(r.judge_llm_output_tokens for r in group_results)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "error": error,
        "skipped": skipped,
        "pass_rate": pass_rate,
        "fail_rate": fail_rate,
        "error_rate": error_rate,
        "skipped_rate": skipped_rate,
        "total_judge_llm_input_tokens": judge_input,
        "total_judge_llm_output_tokens": judge_output,
        "total_judge_llm_tokens": judge_input + judge_output,
    }


def _compute_metric_stats(
    results: list[EvaluationResult],
    compute_ci: bool = False,
) -> dict[str, MetricStats]:
    """Compute per-metric statistics."""
    if not results:
        return {}

    grouped: dict[str, list[EvaluationResult]] = {}
    for r in results:
        grouped.setdefault(r.metric_identifier, []).append(r)

    metric_stats: dict[str, MetricStats] = {}
    for metric_id, group_results in grouped.items():
        base = _build_group_stats_dict(group_results)
        scores = [r.score for r in group_results if r.score is not None]
        base["score_statistics"] = _compute_score_statistics(scores, compute_ci)
        metric_stats[metric_id] = MetricStats(**base)

    return metric_stats


def _compute_conversation_stats(
    results: list[EvaluationResult],
) -> dict[str, ConversationStats]:
    """Compute per-conversation statistics."""
    if not results:
        return {}

    grouped: dict[str, list[EvaluationResult]] = {}
    for r in results:
        grouped.setdefault(r.conversation_group_id, []).append(r)

    conv_stats: dict[str, ConversationStats] = {}
    for conv_id, group_results in grouped.items():
        base = _build_group_stats_dict(group_results)
        conv_stats[conv_id] = ConversationStats(**base)

    return conv_stats


def _compute_tag_stats(
    results: list[EvaluationResult],
    compute_ci: bool = False,
) -> dict[str, TagStats]:
    """Compute per-tag statistics."""
    if not results:
        return {}

    grouped: dict[str, list[EvaluationResult]] = {}
    for r in results:
        grouped.setdefault(r.tag, []).append(r)

    tag_stats: dict[str, TagStats] = {}
    for tag, group_results in grouped.items():
        base = _build_group_stats_dict(group_results)
        scores = [r.score for r in group_results if r.score is not None]
        base["score_statistics"] = _compute_score_statistics(scores, compute_ci)
        tag_stats[tag] = TagStats(**base)

    return tag_stats


def _try_bootstrap(scores: list[float]) -> Optional[dict[str, float]]:
    """Attempt to compute bootstrap confidence intervals for scores.

    Args:
        scores: List of numeric score values.

    Returns:
        Confidence interval dict with low, mean, high, confidence_level, or None.
    """
    try:
        scores_series = pd.Series(scores)
        ci_low, ci_mean, ci_high = bootstrap_intervals(scores_series)
        return {
            "low": float(ci_low),
            "mean": float(ci_mean),
            "high": float(ci_high),
            "confidence_level": 95.0,
        }
    except (ValueError, RuntimeError):
        return None


def _numeric_stats_from_dict(raw: dict[str, Any]) -> Optional[NumericStats]:
    """Convert a raw numeric stats dictionary to a NumericStats model.

    Args:
        raw: Dictionary with count, mean, median, std, min, max keys.

    Returns:
        NumericStats instance, or None if count is 0.
    """
    if raw.get("count", 0) == 0:
        return None

    return NumericStats(
        count=raw["count"],
        mean=raw.get("mean"),
        median=raw.get("median"),
        std=raw.get("std"),
        min_value=raw.get("min"),
        max_value=raw.get("max"),
    )


def _compute_api_token_usage(
    evaluation_data: list[EvaluationData],
) -> ApiTokenUsage:
    """Compute API token usage from evaluation data.

    Args:
        evaluation_data: List of evaluation data with turn-level token counts.

    Returns:
        ApiTokenUsage instance.
    """
    raw = calculate_api_token_usage(evaluation_data)
    return ApiTokenUsage(
        total_api_input_tokens=raw["total_api_input_tokens"],
        total_api_output_tokens=raw["total_api_output_tokens"],
        total_api_tokens=raw["total_api_tokens"],
    )


def _compute_streaming_stats(
    evaluation_data: list[EvaluationData],
) -> Optional[StreamingStats]:
    """Compute streaming statistics from evaluation data.

    Args:
        evaluation_data: List of evaluation data with streaming metrics.

    Returns:
        StreamingStats instance, or None if no streaming data available.
    """
    raw = calculate_streaming_stats(evaluation_data)

    ttft = _numeric_stats_from_dict(raw.get("time_to_first_token", {}))
    duration = _numeric_stats_from_dict(raw.get("streaming_duration", {}))
    throughput = _numeric_stats_from_dict(raw.get("tokens_per_second", {}))

    if ttft is None and duration is None and throughput is None:
        return None

    return StreamingStats(
        time_to_first_token=ttft,
        streaming_duration=duration,
        tokens_per_second=throughput,
    )
