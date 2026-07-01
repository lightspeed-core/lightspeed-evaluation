"""Shared utilities for output and evaluation."""

import statistics
from typing import Optional

import numpy as np
import pandas as pd

from lightspeed_evaluation.core.models.data import (
    EvaluationData,
    EvaluationResult,
)
from lightspeed_evaluation.core.models.statistics import (
    AgentTokenStats,
    AgentTokenUsage,
    ConfidenceInterval,
    ConversationStats,
    DetailedStats,
    MetricStats,
    NumericStats,
    OverallStats,
    ScoreStatistics,
    StreamingStats,
    TagStats,
)


def bootstrap_intervals(
    s: pd.Series, confidence: float = 95, bootstrap_steps: int = 10000
) -> tuple[np.floating, np.floating, np.floating]:
    """Compute confidence interval using bootstraping, return low, mean, high."""
    if not 0 <= confidence <= 100:
        raise ValueError("Invalid confidence, must be between 0 and 100")

    sample_n = len(s)
    sample_mean = np.mean(s)

    confidence_rev = 100 - confidence

    rates = np.array(
        [np.mean(s.sample(n=sample_n, replace=True)) for _ in range(bootstrap_steps)]
    )

    # Median (not mean) is correct here
    mean_boot_strap = np.median(rates)
    low = np.percentile(rates - sample_mean, (confidence_rev / 2.0))
    high = np.percentile(rates - sample_mean, 100 - (confidence_rev / 2.0))

    # high represent lower bound, low represents upper bound
    return sample_mean - high, mean_boot_strap, sample_mean - low


def _try_bootstrap(scores: list[float]) -> Optional[ConfidenceInterval]:
    """Attempt to compute bootstrap confidence intervals for scores."""
    try:
        confidence_level = 95.0
        scores_series = pd.Series(scores)
        ci_low, ci_mean, ci_high = bootstrap_intervals(
            scores_series, confidence=confidence_level
        )
        return ConfidenceInterval(
            low=float(ci_low),
            mean=float(ci_mean),
            high=float(ci_high),
            confidence_level=confidence_level,
        )
    except (ValueError, RuntimeError):
        return None


def compute_numeric_stats(values: list[float]) -> Optional[NumericStats]:
    """Calculate numeric statistics for a list of values."""
    if not values:
        return None

    return NumericStats(
        count=len(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        std=statistics.stdev(values) if len(values) > 1 else 0.0,
        min_value=min(values),
        max_value=max(values),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
    )


def compute_streaming_stats(
    evaluation_data: list[EvaluationData],
) -> Optional[StreamingStats]:
    """Compute streaming performance statistics from evaluation data."""
    ttft_values: list[float] = []
    duration_values: list[float] = []
    throughput_values: list[float] = []

    for conv_data in evaluation_data:
        for turn in conv_data.turns:
            if turn.time_to_first_token is not None:
                ttft_values.append(turn.time_to_first_token)
            if turn.streaming_duration is not None:
                duration_values.append(turn.streaming_duration)
            if turn.tokens_per_second is not None:
                throughput_values.append(turn.tokens_per_second)

    ttft = compute_numeric_stats(ttft_values)
    duration = compute_numeric_stats(duration_values)
    throughput = compute_numeric_stats(throughput_values)

    if ttft is None and duration is None and throughput is None:
        return None

    return StreamingStats(
        time_to_first_token=ttft,
        streaming_duration=duration,
        tokens_per_second=throughput,
    )


def compute_overall_stats(results: list[EvaluationResult]) -> OverallStats:
    """Calculate overall pass/fail/error/skipped/token statistics from results."""
    total = len(results)
    passed = sum(1 for r in results if r.result == "PASS")
    failed = sum(1 for r in results if r.result == "FAIL")
    error = sum(1 for r in results if r.result == "ERROR")
    skipped = sum(1 for r in results if r.result == "SKIPPED")

    total_judge_input = sum(r.judge_llm_input_tokens for r in results)
    total_judge_output = sum(r.judge_llm_output_tokens for r in results)
    total_embedding = sum(r.embedding_tokens for r in results)

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
        total_embedding_tokens=total_embedding,
    )


def compute_score_statistics(
    scores: list[float],
    compute_ci: bool = False,
) -> ScoreStatistics:
    """Compute score statistics from a list of scores."""
    num_stats = compute_numeric_stats(scores)
    if num_stats is None:
        return ScoreStatistics()

    confidence_interval = None
    if compute_ci and len(scores) > 1:
        confidence_interval = _try_bootstrap(scores)

    return ScoreStatistics(
        count=num_stats.count,
        mean=num_stats.mean or 0.0,
        median=num_stats.median or 0.0,
        std=num_stats.std or 0.0,
        min_score=num_stats.min_value or 0.0,
        max_score=num_stats.max_value or 0.0,
        confidence_interval=confidence_interval,
    )


def compute_agent_token_usage(evaluation_data: list[EvaluationData]) -> AgentTokenUsage:
    """Compute agent token usage with totals and statistics from evaluation data."""
    total_input_tokens = 0
    total_output_tokens = 0

    for conv_data in evaluation_data:
        for turn in conv_data.turns:
            total_input_tokens += turn.api_input_tokens
            total_output_tokens += turn.api_output_tokens

    return AgentTokenUsage(
        total_api_input_tokens=total_input_tokens,
        total_api_output_tokens=total_output_tokens,
        total_api_tokens=total_input_tokens + total_output_tokens,
        statistics=compute_agent_token_stats(evaluation_data),
    )


def compute_metric_stats(
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
    for metric_id in sorted(grouped):
        group_results = grouped[metric_id]
        base = compute_overall_stats(group_results)
        scores = [r.score for r in group_results if r.score is not None]
        metric_stats[metric_id] = MetricStats(
            **base.model_dump(),
            score_statistics=compute_score_statistics(scores, compute_ci),
        )

    return metric_stats


def compute_tag_stats(
    results: list[EvaluationResult],
    compute_ci: bool = False,
) -> dict[str, TagStats]:
    """Compute per-tag statistics."""
    if not results:
        return {}

    grouped: dict[str, list[EvaluationResult]] = {}
    for r in results:
        for t in r.tag:
            grouped.setdefault(t, []).append(r)

    tag_stats: dict[str, TagStats] = {}
    for tag in sorted(grouped):
        group_results = grouped[tag]
        base = compute_overall_stats(group_results)
        scores = [r.score for r in group_results if r.score is not None]
        tag_stats[tag] = TagStats(
            **base.model_dump(),
            score_statistics=compute_score_statistics(scores, compute_ci),
        )

    return tag_stats


def compute_conversation_stats(
    results: list[EvaluationResult],
) -> dict[str, ConversationStats]:
    """Compute per-conversation statistics."""
    if not results:
        return {}

    grouped: dict[str, list[EvaluationResult]] = {}
    for r in results:
        grouped.setdefault(r.conversation_group_id, []).append(r)

    conv_stats: dict[str, ConversationStats] = {}
    for conv_id in sorted(grouped):
        group_results = grouped[conv_id]
        base = compute_overall_stats(group_results)
        conv_stats[conv_id] = ConversationStats(**base.model_dump())

    return conv_stats


def compute_detailed_stats(results: list[EvaluationResult]) -> DetailedStats:
    """Calculate detailed statistics broken down by different categories."""
    if not results:
        return DetailedStats(by_metric={}, by_conversation={}, by_tag={})

    return DetailedStats(
        by_metric=compute_metric_stats(results, compute_ci=True),
        by_conversation=compute_conversation_stats(results),
        by_tag=compute_tag_stats(results, compute_ci=True),
    )


def compute_field_numeric_stats_from_evaluation_data(
    evaluation_data: list[EvaluationData], field_name: str
) -> Optional[NumericStats]:
    """Calculate statistics for a numeric field, filtering out zeros (unmeasured values).

    Args:
        evaluation_data: List of evaluation records to inspect.
        field_name: Name of the numeric field to compute stats for.

    Returns:
        Optional[NumericStats]: Dictionary of computed statistics including count, mean, median,
        min, max, and standard deviation. Note that zero values are filtered out before
        computing statistics as they represent unmeasured values.
    """
    values = []
    for conv_data in evaluation_data:
        for turn in conv_data.turns:
            value = getattr(turn, field_name, 0)
            if value > 0:
                values.append(value)

    return compute_numeric_stats(values)


def compute_agent_latency_stats(
    evaluation_data: list[EvaluationData],
) -> Optional[NumericStats]:
    """Compute agent latency statistics from evaluation data.

    Args:
        evaluation_data: List of evaluation data containing turn-level latency values.

    Returns:
        NumericStats instance with computed statistics, or None if no evaluation data
        or no valid (non-zero) latency values exist.
    """
    if not evaluation_data:
        return None
    num_stats = compute_field_numeric_stats_from_evaluation_data(
        evaluation_data, "agent_latency"
    )
    return num_stats


def compute_agent_token_stats(
    evaluation_data: list[EvaluationData],
) -> Optional[AgentTokenStats]:
    """Calculate agent token usage statistics with percentiles from evaluation data.

    Args:
        evaluation_data: List of evaluation data containing turn-level token counts.

    Returns:
        AgentTokenStats instance with input/output token statistics, or None if no data.
    """
    if not evaluation_data:
        return None

    input_tokens_stats = compute_field_numeric_stats_from_evaluation_data(
        evaluation_data, "api_input_tokens"
    )
    output_tokens_stats = compute_field_numeric_stats_from_evaluation_data(
        evaluation_data, "api_output_tokens"
    )

    return AgentTokenStats(input=input_tokens_stats, output=output_tokens_stats)
