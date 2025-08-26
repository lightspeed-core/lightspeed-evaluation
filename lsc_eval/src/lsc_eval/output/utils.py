"""Shared utilities for output and evaluation."""

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..core.models import EvaluationResult, TurnData


@dataclass
class EvaluationScope:
    """Scope and parameters for metric evaluation."""

    turn_idx: Optional[int] = None
    turn_data: Optional[TurnData] = None
    is_conversation: bool = False


def calculate_basic_stats(results: List[EvaluationResult]) -> Dict[str, Any]:
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
        }

    total = len(results)
    pass_count = sum(1 for r in results if r.result == "PASS")
    fail_count = sum(1 for r in results if r.result == "FAIL")
    error_count = sum(1 for r in results if r.result == "ERROR")

    return {
        "TOTAL": total,
        "PASS": pass_count,
        "FAIL": fail_count,
        "ERROR": error_count,
        "pass_rate": (pass_count / total) * 100 if total > 0 else 0,
        "fail_rate": (fail_count / total) * 100 if total > 0 else 0,
        "error_rate": (error_count / total) * 100 if total > 0 else 0,
    }


def calculate_detailed_stats(results: List[EvaluationResult]) -> Dict[str, Any]:
    """Calculate detailed statistics broken down by metric and conversation."""
    if not results:
        return {"by_metric": {}, "by_conversation": {}}

    by_metric: Dict[str, Dict[str, Any]] = {}
    by_conversation: Dict[str, Dict[str, Any]] = {}

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


def _update_metric_stats(by_metric: Dict[str, Dict[str, Any]], result: EvaluationResult) -> None:
    """Update metric statistics with a single result."""
    if result.metric_identifier not in by_metric:
        by_metric[result.metric_identifier] = {"pass": 0, "fail": 0, "error": 0, "scores": []}

    stats = by_metric[result.metric_identifier]
    stats[result.result.lower()] += 1

    if result.score is not None:
        stats["scores"].append(result.score)


def _update_conversation_stats(
    by_conversation: Dict[str, Dict[str, Any]], result: EvaluationResult
) -> None:
    """Update conversation statistics with a single result."""
    if result.conversation_group_id not in by_conversation:
        by_conversation[result.conversation_group_id] = {"pass": 0, "fail": 0, "error": 0}

    by_conversation[result.conversation_group_id][result.result.lower()] += 1


def _finalize_metric_stats(stats: Dict[str, Any]) -> None:
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
        stats["score_statistics"] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "count": len(scores),
        }
    else:
        stats["score_statistics"] = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
        }


def _finalize_conversation_stats(stats: Dict[str, Any]) -> None:
    """Calculate final statistics for a conversation."""
    total = stats["pass"] + stats["fail"] + stats["error"]
    if total > 0:
        stats["pass_rate"] = stats["pass"] / total * 100
        stats["fail_rate"] = stats["fail"] / total * 100
        stats["error_rate"] = stats["error"] / total * 100
    else:
        stats["pass_rate"] = 0.0
        stats["fail_rate"] = 0.0
        stats["error_rate"] = 0.0
