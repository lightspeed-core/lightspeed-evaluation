"""Serialization functions for converting statistics models to JSON-compatible dicts."""

from typing import Any

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.models.statistics import (
    ConversationStats,
    MetricStats,
    NumericStats,
    OverallStats,
    StreamingStats,
    TagStats,
)
from lightspeed_evaluation.core.models.summary import (
    EvaluationSummary,
)


def result_to_json_dict(r: EvaluationResult) -> dict[str, Any]:
    """Convert a single EvaluationResult to JSON-serializable dict.

    Args:
        r: The evaluation result to convert.

    Returns:
        Dictionary matching the existing JSON summary result format.
    """
    return {
        "conversation_group_id": r.conversation_group_id,
        "tag": r.tag,
        "turn_id": r.turn_id,
        "metric_identifier": r.metric_identifier,
        "result": r.result,
        "score": r.score,
        "threshold": r.threshold,
        "execution_time": r.execution_time,
        "evaluation_latency": r.evaluation_latency,
        "judge_llm_input_tokens": r.judge_llm_input_tokens,
        "judge_llm_output_tokens": r.judge_llm_output_tokens,
        "judge_scores": (
            [js.model_dump() for js in r.judge_scores] if r.judge_scores else None
        ),
        "time_to_first_token": r.time_to_first_token,
        "streaming_duration": r.streaming_duration,
        "agent_latency": r.agent_latency,
        "tokens_per_second": r.tokens_per_second,
    }


def overall_to_basic_stats_dict(
    overall: OverallStats,
) -> dict[str, Any]:
    """Convert OverallStats to the dict format expected by text output.

    Args:
        overall: OverallStats model instance.

    Returns:
        Dictionary with keys matching the original calculate_basic_stats format.
    """
    return {
        "TOTAL": overall.total,
        "PASS": overall.passed,
        "FAIL": overall.failed,
        "ERROR": overall.error,
        "SKIPPED": overall.skipped,
        "pass_rate": overall.pass_rate,
        "fail_rate": overall.fail_rate,
        "error_rate": overall.error_rate,
        "skipped_rate": overall.skipped_rate,
        "total_judge_llm_input_tokens": overall.total_judge_llm_input_tokens,
        "total_judge_llm_output_tokens": overall.total_judge_llm_output_tokens,
        "total_judge_llm_tokens": overall.total_judge_llm_tokens,
        "total_embedding_tokens": overall.total_embedding_tokens,
    }


def _group_stats_to_dict(
    stats: MetricStats | ConversationStats | TagStats,
) -> dict[str, Any]:
    """Convert a group stats model to the dict format for text output.

    Args:
        stats: MetricStats, ConversationStats, or TagStats instance.

    Returns:
        Dictionary with lowercase keys matching original detailed stats format.
    """
    result: dict[str, Any] = {
        "pass": stats.passed,
        "fail": stats.failed,
        "error": stats.error,
        "skipped": stats.skipped,
        "pass_rate": stats.pass_rate,
        "fail_rate": stats.fail_rate,
        "error_rate": stats.error_rate,
        "skipped_rate": stats.skipped_rate,
    }
    if (
        isinstance(stats, (MetricStats, TagStats))
        and stats.score_statistics is not None
    ):
        score_stats = stats.score_statistics
        result["score_statistics"] = {
            "count": score_stats.count,
            "mean": score_stats.mean,
            "median": score_stats.median,
            "std": score_stats.std,
            "min": score_stats.min_score,
            "max": score_stats.max_score,
            "confidence_interval": (
                score_stats.confidence_interval.model_dump()
                if score_stats.confidence_interval is not None
                else None
            ),
        }
    return result


def metric_stats_to_dict(
    by_metric: dict[str, MetricStats],
) -> dict[str, dict[str, Any]]:
    """Convert by_metric model dict to legacy dict format.

    Args:
        by_metric: Dictionary mapping metric IDs to MetricStats models.

    Returns:
        Dictionary in the original detailed stats format.
    """
    return {k: _group_stats_to_dict(v) for k, v in by_metric.items()}


def conversation_stats_to_dict(
    by_conversation: dict[str, ConversationStats],
) -> dict[str, dict[str, Any]]:
    """Convert by_conversation model dict to legacy dict format.

    Args:
        by_conversation: Dictionary mapping conversation IDs to ConversationStats.

    Returns:
        Dictionary in the original detailed stats format.
    """
    return {k: _group_stats_to_dict(v) for k, v in by_conversation.items()}


def tag_stats_to_dict(
    by_tag: dict[str, TagStats],
) -> dict[str, dict[str, Any]]:
    """Convert by_tag model dict to legacy dict format.

    Args:
        by_tag: Dictionary mapping tags to TagStats models.

    Returns:
        Dictionary in the original detailed stats format.
    """
    return {k: _group_stats_to_dict(v) for k, v in by_tag.items()}


def summary_to_detailed_stats_dict(
    summary: EvaluationSummary,
) -> dict[str, Any]:
    """Convert EvaluationSummary to the detailed stats dict format.

    This produces a dictionary with by_metric, by_conversation, by_tag keys
    matching the format from compute_detailed_stats().

    Args:
        summary: The EvaluationSummary instance.

    Returns:
        Dictionary matching the original detailed stats format.
    """
    return {
        "by_metric": metric_stats_to_dict(summary.by_metric),
        "by_conversation": conversation_stats_to_dict(summary.by_conversation),
        "by_tag": tag_stats_to_dict(summary.by_tag),
    }


def streaming_stats_to_dict(streaming: StreamingStats) -> dict[str, Any]:
    """Convert StreamingStats model to the dict format for text output.

    Args:
        streaming: StreamingStats model instance.

    Returns:
        Dictionary matching the original streaming stats format.
    """
    result: dict[str, Any] = {}
    for field_name in (
        "time_to_first_token",
        "streaming_duration",
        "tokens_per_second",
    ):
        numeric = getattr(streaming, field_name, None)
        if numeric is not None:
            result[field_name] = numeric_stats_to_dict(numeric)
        else:
            result[field_name] = {"count": 0}
    return result


def numeric_stats_to_dict(numeric: NumericStats) -> dict[str, Any]:
    """Convert NumericStats model to dict format for text output."""
    return {
        "count": numeric.count,
        "mean": numeric.mean,
        "median": numeric.median,
        "std": numeric.std,
        "min": numeric.min_value,
        "max": numeric.max_value,
        "p95": numeric.p95,
        "p99": numeric.p99,
    }
