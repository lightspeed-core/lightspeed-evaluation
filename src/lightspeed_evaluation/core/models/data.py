"""Core data models for evaluation framework."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightspeed_evaluation.core.constants import (
    MAX_RUN_NAME_LENGTH,
    SUPPORTED_RESULT_STATUSES,
)

logger = logging.getLogger(__name__)


def _validate_and_deduplicate_metrics(
    metrics: list[str], metric_type: str = "metric"
) -> list[str]:
    """Validate format and deduplicate metrics while preserving order."""
    # Validate format first
    for metric in metrics:
        if not metric or ":" not in metric:
            raise ValueError(
                f'{metric_type} "{metric}" must be in format "framework:metric_name"'
            )

    # Deduplicate while preserving order
    seen = set()
    deduplicated = []
    for metric in metrics:
        if metric not in seen:
            deduplicated.append(metric)
            seen.add(metric)
    return deduplicated


class TurnData(BaseModel):
    """Individual turn data within a conversation."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(..., min_length=1, description="Turn ID (alphanumeric)")
    query: str = Field(..., min_length=1, description="Query")
    attachments: Optional[list[str]] = Field(
        default=None, min_length=0, description="Attachments"
    )
    response: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Actual Response - populated by API if enabled",
    )
    tool_calls: Optional[list[list[dict[str, Any]]]] = Field(
        default=None, description="Actual Tool calls - populated by API if enabled"
    )
    contexts: Optional[list[str]] = Field(
        default=None, min_length=1, description="Contexts"
    )
    expected_response: Optional[str] = Field(
        default=None, min_length=1, description="Expected response for comparison"
    )
    expected_tool_calls: Optional[list[list[dict[str, Any]]]] = Field(
        default=None, description="Expected tool call sequences"
    )
    expected_intent: Optional[str] = Field(
        default=None, min_length=1, description="Expected intent for intent evaluation"
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID - populated by API if enabled"
    )

    # Per-turn metrics support
    turn_metrics: Optional[list[str]] = Field(
        default=None,
        description="Turn-specific metrics to evaluate (overrides system defaults)",
    )
    turn_metrics_metadata: Optional[dict[str, dict[str, Any]]] = Field(
        default=None,
        description="Turn-specific metric configuration (overrides system defaults)",
    )

    # Script execution support
    verify_script: Optional[Union[str, Path]] = Field(
        default=None, description="Path to verify script for script-based evaluation"
    )

    @field_validator("turn_metrics")
    @classmethod
    def validate_turn_metrics(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate and deduplicate turn-specific metrics."""
        if v is not None:
            v = _validate_and_deduplicate_metrics(v, "Turn metric")
        return v

    @field_validator("expected_tool_calls", mode="before")
    @classmethod
    def validate_expected_tool_calls(
        cls, v: Optional[Any]
    ) -> Optional[list[list[dict[str, Any]]]]:
        """Validate expected tool calls when provided."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("Expected tool calls must be a list of sequences")

        result = []
        for i, sequence in enumerate(v):
            if not isinstance(sequence, list):
                raise ValueError(f"Sequence {i} must be a list")

            tool_calls = []
            for j, tool_call in enumerate(sequence):
                if not isinstance(tool_call, dict):
                    raise ValueError(
                        f"Tool call {j} in sequence {i} must be a dictionary"
                    )

                # Validate required keys
                if "tool_name" not in tool_call:
                    raise ValueError(
                        f"Tool call {j} in sequence {i} missing required 'tool_name' field"
                    )

                if not tool_call["tool_name"]:
                    raise ValueError(
                        f"Tool call {j} in sequence {i} has empty 'tool_name' field"
                    )

                # Ensure arguments field exists (can be empty dict)
                validated_tool_call = {
                    "tool_name": tool_call["tool_name"],
                    "arguments": tool_call.get("arguments", {}),
                }
                tool_calls.append(validated_tool_call)

            result.append(tool_calls)
        return result


class EvaluationData(BaseModel):
    """Complete evaluation data for a conversation group."""

    model_config = ConfigDict(extra="forbid")

    conversation_group_id: str = Field(
        ..., min_length=1, description="Unique conversation group identifier"
    )
    description: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional description of the conversation group",
    )
    run_name: Optional[str] = Field(
        default=None,
        max_length=MAX_RUN_NAME_LENGTH,
        description=f"Optional name for this evaluation run (max {MAX_RUN_NAME_LENGTH} chars)",
    )

    # Conversation-level metrics
    conversation_metrics: Optional[list[str]] = Field(
        default=None, description="Conversation-level metrics to evaluate"
    )

    # Conversation-level metric configuration
    conversation_metrics_metadata: Optional[dict[str, dict[str, Any]]] = Field(
        default=None, description="Conversation-level metric configuration"
    )

    # Conversation turns
    turns: list[TurnData] = Field(
        ..., min_length=1, description="Conversation turns - must have at least one"
    )

    # Script execution support
    setup_script: Optional[Union[str, Path]] = Field(
        default=None,
        description="Path to setup script to run before conversation starts",
    )
    cleanup_script: Optional[Union[str, Path]] = Field(
        default=None,
        description="Path to cleanup script to run after conversation ends",
    )

    @field_validator("conversation_metrics")
    @classmethod
    def validate_conversation_metrics(
        cls, v: Optional[list[str]]
    ) -> Optional[list[str]]:
        """Validate and deduplicate conversation metrics."""
        if v is not None:
            v = _validate_and_deduplicate_metrics(v, "Conversation metric")
        return v


class EvaluationResult(BaseModel):
    """Single evaluation result."""

    model_config = ConfigDict(extra="forbid")

    conversation_group_id: str = Field(
        ..., min_length=1, description="Conversation group identifier"
    )
    turn_id: Optional[str] = Field(
        default=None, description="Turn ID if turn-level evaluation"
    )
    metric_identifier: str = Field(
        ...,
        min_length=1,
        description="Metric identifier (e.g., 'ragas:response_relevancy')",
    )
    result: str = Field(..., description="Evaluation result status")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Score between 0 and 1"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold for pass/fail",
    )
    reason: str = Field(default="", description="Explanation of the result")
    query: str = Field(default="", description="Query text")
    response: str = Field(default="", description="Response text")
    execution_time: float = Field(
        default=0, ge=0, description="Execution time in seconds"
    )

    @field_validator("result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        """Validate result is valid status."""
        if v not in SUPPORTED_RESULT_STATUSES:
            raise ValueError(f"Result must be one of {SUPPORTED_RESULT_STATUSES}")
        return v


class EvaluationScope(BaseModel):
    """Scope and parameters for metric evaluation."""

    turn_idx: Optional[int] = Field(
        default=None, ge=0, description="Turn index for turn-level evaluation"
    )
    turn_data: Optional[TurnData] = Field(
        default=None, description="Turn data for turn-level evaluation"
    )
    is_conversation: bool = Field(
        default=False, description="Whether this is conversation-level evaluation"
    )


class EvaluationRequest(BaseModel):
    """Evaluation request data model with validation."""

    conv_data: EvaluationData = Field(..., description="Conversation data")
    metric_identifier: str = Field(
        ..., min_length=1, description="Metric identifier (e.g., 'ragas:faithfulness')"
    )
    is_conversation: bool = Field(
        default=False, description="Whether this is conversation-level evaluation"
    )
    turn_idx: Optional[int] = Field(
        default=None, ge=0, description="Turn index for turn-level evaluation"
    )
    turn_data: Optional[TurnData] = Field(
        default=None, description="Turn data for turn-level evaluation"
    )

    # Computed field for convenience
    turn_id: Optional[str] = Field(
        default=None, description="Turn ID extracted from turn_data"
    )

    def model_post_init(self, context: Any, /) -> None:
        """Post-initialization to set computed fields."""
        if self.turn_data:
            self.turn_id = self.turn_data.turn_id  # pylint: disable=no-member

    @classmethod
    def for_turn(
        cls,
        conv_data: EvaluationData,
        metric_identifier: str,
        turn_idx: int,
        turn_data: TurnData,
    ) -> "EvaluationRequest":
        """Create request for turn-level evaluation."""
        return cls(
            conv_data=conv_data,
            metric_identifier=metric_identifier,
            is_conversation=False,
            turn_idx=turn_idx,
            turn_data=turn_data,
        )

    @classmethod
    def for_conversation(
        cls, conv_data: EvaluationData, metric_identifier: str
    ) -> "EvaluationRequest":
        """Create request for conversation-level evaluation."""
        return cls(
            conv_data=conv_data,
            metric_identifier=metric_identifier,
            is_conversation=True,
        )
