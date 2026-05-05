"""Pydantic models for evaluation statistics."""

from typing import Optional
from pydantic import BaseModel, Field


class NumericStats(BaseModel):
    """Numeric statistics for a set of values (e.g., TTFT, duration, latency)."""

    count: int = Field(default=0, description="Number of values")
    mean: Optional[float] = Field(default=None, description="Mean value")
    median: Optional[float] = Field(default=None, description="Median value")
    std: Optional[float] = Field(default=None, description="Standard deviation")
    min_value: Optional[float] = Field(default=None, description="Minimum value")
    max_value: Optional[float] = Field(default=None, description="Maximum value")
    p95: Optional[float] = Field(default=None, description="95th percentile")
    p99: Optional[float] = Field(default=None, description="99th percentile")


class ConfidenceInterval(BaseModel):
    """Bootstrap confidence interval for score statistics."""

    low: float = Field(
        default=0.0, description="Lower bound of the confidence interval"
    )
    mean: float = Field(default=0.0, description="Mean value from bootstrap samples")
    high: float = Field(
        default=0.0, description="Upper bound of the confidence interval"
    )
    confidence_level: float = Field(
        default=95.0,
        description="Confidence level as a percentage (e.g., 95.0 for 95%)",
    )


class ScoreStatistics(BaseModel):
    """Score statistics for a metric or group."""

    count: int = Field(default=0, description="Number of scored results")
    mean: float = Field(default=0.0, description="Mean score")
    median: float = Field(default=0.0, description="Median score")
    std: float = Field(default=0.0, description="Standard deviation")
    min_score: float = Field(default=0.0, description="Minimum score")
    max_score: float = Field(default=0.0, description="Maximum score")
    confidence_interval: Optional[ConfidenceInterval] = Field(
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
    total_embedding_tokens: int = Field(default=0, description="Total embedding tokens")


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


class DetailedStats(BaseModel):
    """Detailed statistics broken down by metric, conversation, and tag."""

    by_metric: dict[str, MetricStats] = Field(
        default_factory=dict, description="Statistics grouped by metric name"
    )
    by_conversation: dict[str, ConversationStats] = Field(
        default_factory=dict, description="Statistics grouped by conversation ID"
    )
    by_tag: dict[str, TagStats] = Field(
        default_factory=dict, description="Statistics grouped by tag"
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


class AgentTokenStats(BaseModel):
    """Agent token usage statistics with percentiles."""

    input: Optional[NumericStats] = Field(
        default=None, description="Input token statistics"
    )
    output: Optional[NumericStats] = Field(
        default=None, description="Output token statistics"
    )


class AgentTokenUsage(BaseModel):
    """Agent token usage totals and statistics."""

    total_api_input_tokens: int = Field(default=0, description="Total API input tokens")
    total_api_output_tokens: int = Field(
        default=0, description="Total API output tokens"
    )
    total_api_tokens: int = Field(default=0, description="Total API tokens")
    statistics: Optional[AgentTokenStats] = Field(
        default=None, description="Agent token usage statistics with percentiles"
    )
