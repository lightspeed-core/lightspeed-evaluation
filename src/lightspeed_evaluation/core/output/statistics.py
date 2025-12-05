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
    """Calculate basic pass/fail/error statistics from results."""
    if not results:
        return {
            "TOTAL": 0,
            "PASS": 0,
            "FAIL": 0,
            "ERROR": 0,
            "pass_rate": 0.0,
            "fail_rate": 0.0,
            "error_rate": 0.0,
            "total_judge_llm_input_tokens": 0,
            "total_judge_llm_output_tokens": 0,
            "total_judge_llm_tokens": 0,
        }

    total = len(results)
    pass_count = sum(1 for r in results if r.result == "PASS")
    fail_count = sum(1 for r in results if r.result == "FAIL")
    error_count = sum(1 for r in results if r.result == "ERROR")

    # Calculate token totals
    total_judge_input = sum(r.judge_llm_input_tokens for r in results)
    total_judge_output = sum(r.judge_llm_output_tokens for r in results)

    return {
        "TOTAL": total,
        "PASS": pass_count,
        "FAIL": fail_count,
        "ERROR": error_count,
        "pass_rate": (pass_count / total) * 100 if total > 0 else 0,
        "fail_rate": (fail_count / total) * 100 if total > 0 else 0,
        "error_rate": (error_count / total) * 100 if total > 0 else 0,
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

    # Collect data
    for result in results:
        _update_metric_stats(by_metric, result)
        _update_conversation_stats(by_conversation, result)

    # Calculate derived statistics
    for _, stats in by_metric.items():
        _finalize_metric_stats(stats)

    for _, stats in by_conversation.items():
        _finalize_conversation_stats(stats)

    return {"by_metric": by_metric, "by_conversation": by_conversation}


def _update_metric_stats(
    by_metric: dict[str, dict[str, Any]], result: EvaluationResult
) -> None:
    """Update metric statistics with a single result."""
    if result.metric_identifier not in by_metric:
        by_metric[result.metric_identifier] = {
            "pass": 0,
            "fail": 0,
            "error": 0,
            "scores": [],
        }

    stats = by_metric[result.metric_identifier]
    stats[result.result.lower()] += 1

    if result.score is not None:
        stats["scores"].append(result.score)


def _update_conversation_stats(
    by_conversation: dict[str, dict[str, Any]], result: EvaluationResult
) -> None:
    """Update conversation statistics with a single result."""
    if result.conversation_group_id not in by_conversation:
        by_conversation[result.conversation_group_id] = {
            "pass": 0,
            "fail": 0,
            "error": 0,
        }

    by_conversation[result.conversation_group_id][result.result.lower()] += 1


def _finalize_metric_stats(stats: dict[str, Any]) -> None:
    """Calculate final statistics for a metric."""
    total = stats["pass"] + stats["fail"] + stats["error"]
    if total > 0:
        stats["pass_rate"] = stats["pass"] / total * 100
        stats["fail_rate"] = stats["fail"] / total * 100
        stats["error_rate"] = stats["error"] / total * 100
    else:
        stats["pass_rate"] = 0.0
        stats["fail_rate"] = 0.0
        stats["error_rate"] = 0.0

    # Calculate statistical measures for scores
    if stats["scores"]:
        scores = stats["scores"]
        scores_series = pd.Series(scores)

        # Calculate basic statistics
        score_stats = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }

        # Calculate confidence intervals using bootstrap
        if len(scores) > 1:  # Need at least 2 samples for meaningful bootstrap
            try:
                ci_low, ci_mean, ci_high = bootstrap_intervals(scores_series)
                score_stats["confidence_interval"] = {
                    "low": float(ci_low),
                    "mean": float(ci_mean),
                    "high": float(ci_high),
                    "confidence_level": 95,  # Default confidence level
                }
            except (ValueError, RuntimeError):
                # If bootstrap fails, set confidence interval to None
                score_stats["confidence_interval"] = None
        else:
            score_stats["confidence_interval"] = None

        stats["score_statistics"] = score_stats
    else:
        stats["score_statistics"] = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
            "confidence_interval": None,
        }


def _finalize_conversation_stats(stats: dict[str, Any]) -> None:
    """Calculate final statistics for a conversation."""
    total = stats["pass"] + stats["fail"] + stats["error"]
    if total > 0:
        stats["pass_rate"] = stats["pass"] / total * 100
        stats["fail_rate"] = stats["fail"] / total * 100
        stats["error_rate"] = stats["error"] / total * 100

        # Calculate confidence intervals for conversation rates
        if total > 1:  # Need at least 2 samples for meaningful bootstrap
            try:
                # Create binary series for each outcome type
                # Note: We need to reconstruct the original results for this conversation
                # Since we don't have access to the original results here,
                # we'll skip CI for conversations. This could be enhanced by
                # passing the original results to this function
                stats["confidence_intervals"] = None
            except (ValueError, RuntimeError):
                stats["confidence_intervals"] = None
        else:
            stats["confidence_intervals"] = None
    else:
        stats["pass_rate"] = 0.0
        stats["fail_rate"] = 0.0
        stats["error_rate"] = 0.0
        stats["confidence_intervals"] = None


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
