"""Evaluation summary models for structured results."""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field

from lightspeed_evaluation.core.models.data import (
    EvaluationData,
    EvaluationResult,
)
from lightspeed_evaluation.core.models.statistics import (
    AgentTokenUsage,
    ConversationStats,
    MetricStats,
    NumericStats,
    OverallStats,
    StreamingStats,
    TagStats,
)
from lightspeed_evaluation.core.output.statistics import (
    compute_agent_latency_stats,
    compute_agent_token_usage,
    compute_conversation_stats,
    compute_metric_stats,
    compute_overall_stats,
    compute_streaming_stats,
    compute_tag_stats,
)


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
    agent_token_usage: Optional[AgentTokenUsage] = Field(
        default=None, description="Agent token usage with totals and statistics"
    )
    agent_latency_stats: Optional[NumericStats] = Field(
        default=None, description="Agent latency statistics (when API enabled)"
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

        Args:
            results: List of evaluation results to summarize.
            evaluation_data: Optional evaluation data for API token, agent latency,
                        and streaming stats.
            compute_confidence_intervals: Whether to compute bootstrap confidence
                intervals. Default False.

        Returns:
            A fully populated EvaluationSummary instance.
        """
        timestamp = datetime.now(UTC).isoformat()

        # Compute overall stats
        overall = compute_overall_stats(results)

        # Compute per-metric, per-conversation, per-tag stats
        by_metric = compute_metric_stats(results, compute_confidence_intervals)
        by_conversation = compute_conversation_stats(results)
        by_tag = compute_tag_stats(results, compute_confidence_intervals)

        # Compute API token usage and streaming stats if evaluation data provided
        streaming = None
        agent_token_usage = None
        agent_latency_stats = None
        if evaluation_data:
            streaming = compute_streaming_stats(evaluation_data)
            agent_token_usage = compute_agent_token_usage(evaluation_data)
            agent_latency_stats = compute_agent_latency_stats(evaluation_data)

        return cls(
            timestamp=timestamp,
            results=results,
            overall=overall,
            by_metric=by_metric,
            by_conversation=by_conversation,
            by_tag=by_tag,
            agent_token_usage=agent_token_usage,
            agent_latency_stats=agent_latency_stats,
            streaming=streaming,
        )
