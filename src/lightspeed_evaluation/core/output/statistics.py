"""Shared utilities for output and evaluation."""

import statistics
from typing import Any

import numpy as np
import pandas as pd

from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult


def bootstrap_intervals(
    s: pd.Series, confidence: int = 95, bootstrap_steps: int = 10000
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


def calculate_basic_stats(results: list[EvaluationResult]) -> dict[str, Any]:
    """Calculate basic pass/fail/error/skipped statistics from results."""
    if not results:
        return {
            "TOTAL": 0,
            "PASS": 0,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "error_rate": 0.0,
            "skipped_rate": 0.0,
            "total_judge_llm_input_tokens": 0,
            "total_judge_llm_output_tokens": 0,
            "total_judge_llm_tokens": 0,
        }

    total = len(results)
    pass_count = sum(1 for r in results if r.result == "PASS")
    fail_count = sum(1 for r in results if r.result == "FAIL")
    error_count = sum(1 for r in results if r.result == "ERROR")
    skipped_count = sum(1 for r in results if r.result == "SKIPPED")

    # Calculate token totals
    total_judge_input = sum(r.judge_llm_input_tokens for r in results)
    total_judge_output = sum(r.judge_llm_output_tokens for r in results)

    return {
        "TOTAL": total,
        "PASS": pass_count,
        "FAIL": fail_count,
        "ERROR": error_count,
        "SKIPPED": skipped_count,
        "pass_rate": (pass_count / total) * 100 if total > 0 else 0,
        "fail_rate": (fail_count / total) * 100 if total > 0 else 0,
        "error_rate": (error_count / total) * 100 if total > 0 else 0,
        "skipped_rate": (skipped_count / total) * 100 if total > 0 else 0,
        "total_judge_llm_input_tokens": total_judge_input,
        "total_judge_llm_output_tokens": total_judge_output,
        "total_judge_llm_tokens": total_judge_input + total_judge_output,
    }


def calculate_detailed_stats(results: list[EvaluationResult]) -> dict[str, Any]:
    """Calculate detailed statistics broken down by metric and conversation."""
    if not results:
        return {"by_metric": {}, "by_conversation": {}}

    by_metric: dict[str, dict[str, Any]] = {}
    by_conversation: dict[str, dict[str, Any]] = {}

    # Collect data using generic update function
    for result in results:
        _update_stats(by_metric, result.metric_identifier, result, include_scores=True)
        _update_stats(by_conversation, result.conversation_group_id, result)

    # Finalize statistics for each group
    for stats in by_metric.values():
        _finalize_group_stats(stats, include_scores=True)

    # Note: Conversations don't include score_statistics with confidence intervals.
    # To calculate CI for conversations, we would need to reconstruct the original
    # results for each conversation to create binary series for each outcome type.
    # This could be enhanced by passing the original results to this function.
    for stats in by_conversation.values():
        _finalize_group_stats(stats)

    return {"by_metric": by_metric, "by_conversation": by_conversation}


def _create_empty_stats(*, include_scores: bool = False) -> dict[str, Any]:
    """Create empty statistics dictionary.

    Args:
        include_scores: Whether to include a scores list for score tracking.
    """
    stats: dict[str, Any] = {
        "pass": 0,
        "fail": 0,
        "error": 0,
        "skipped": 0,
    }
    if include_scores:
        stats["scores"] = []
    return stats


def _update_stats(
    stats_dict: dict[str, dict[str, Any]],
    key: str,
    result: EvaluationResult,
    *,
    include_scores: bool = False,
) -> None:
    """Update statistics dictionary with a result.

    Args:
        stats_dict: Dictionary mapping keys to their statistics.
        key: The key to update (e.g., metric_identifier, conversation_group_id).
        result: The evaluation result to add.
        include_scores: Whether to track individual scores.
    """
    if key not in stats_dict:
        stats_dict[key] = _create_empty_stats(include_scores=include_scores)

    stats = stats_dict[key]
    stats[result.result.lower()] += 1

    if include_scores and result.score is not None:
        stats["scores"].append(result.score)


def _calculate_rates(stats: dict[str, Any]) -> None:
    """Calculate pass/fail/error/skipped rates for a stats dictionary."""
    total = stats["pass"] + stats["fail"] + stats["error"] + stats["skipped"]
    if total > 0:
        stats["pass_rate"] = stats["pass"] / total * 100
        stats["fail_rate"] = stats["fail"] / total * 100
        stats["error_rate"] = stats["error"] / total * 100
        stats["skipped_rate"] = stats["skipped"] / total * 100
    else:
        stats["pass_rate"] = 0.0
        stats["fail_rate"] = 0.0
        stats["error_rate"] = 0.0
        stats["skipped_rate"] = 0.0


def _calculate_numeric_stats(values: list[float]) -> dict[str, Any]:
    """Calculate basic numeric statistics for a list of values.

    Args:
        values: List of numeric values.

    Returns:
        Dictionary containing count, mean, median, std, min, max.
    """
    if not values:
        return {"count": 0}

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _calculate_score_statistics(scores: list[float]) -> dict[str, Any]:
    """Calculate score statistics with confidence intervals.

    Args:
        scores: List of score values.

    Returns:
        Dictionary containing mean, median, std, min, max, count, and confidence_interval.
    """
    if not scores:
        return {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
            "confidence_interval": None,
        }

    score_stats = _calculate_numeric_stats(scores)
    score_stats["confidence_interval"] = None

    # Calculate confidence intervals using bootstrap
    if len(scores) > 1:  # Need at least 2 samples for meaningful bootstrap
        try:
            scores_series = pd.Series(scores)
            ci_low, ci_mean, ci_high = bootstrap_intervals(scores_series)
            score_stats["confidence_interval"] = {
                "low": float(ci_low),
                "mean": float(ci_mean),
                "high": float(ci_high),
                "confidence_level": 95,
            }
        except (ValueError, RuntimeError):
            pass  # confidence_interval already set to None

    return score_stats


def _finalize_group_stats(
    stats: dict[str, Any], *, include_scores: bool = False
) -> None:
    """Finalize statistics for a group (calculate rates and optionally score stats).

    Args:
        stats: Statistics dictionary to finalize.
        include_scores: Whether to calculate score statistics.
    """
    _calculate_rates(stats)
    if include_scores:
        stats["score_statistics"] = _calculate_score_statistics(stats.get("scores", []))


def calculate_api_token_usage(evaluation_data: list[EvaluationData]) -> dict[str, Any]:
    """Calculate total API token usage from evaluation data.

    Args:
        evaluation_data: List of evaluation data containing turn-level API token counts.

    Returns:
        Dictionary containing total_api_input_tokens, total_api_output_tokens,
        and total_api_tokens.
    """
    total_input_tokens = 0
    total_output_tokens = 0

    for conv_data in evaluation_data:
        for turn in conv_data.turns:
            total_input_tokens += turn.api_input_tokens
            total_output_tokens += turn.api_output_tokens

    return {
        "total_api_input_tokens": total_input_tokens,
        "total_api_output_tokens": total_output_tokens,
        "total_api_tokens": total_input_tokens + total_output_tokens,
    }


def calculate_streaming_stats(
    evaluation_data: list[EvaluationData],
) -> dict[str, Any]:
    """Calculate streaming performance statistics from evaluation data.

    Args:
        evaluation_data: List of evaluation data containing turn-level streaming metrics.

    Returns:
        Dictionary containing streaming performance statistics (TTFT, duration, throughput).
    """
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

    return {
        "time_to_first_token": _calculate_numeric_stats(ttft_values),
        "streaming_duration": _calculate_numeric_stats(duration_values),
        "tokens_per_second": _calculate_numeric_stats(throughput_values),
    }
