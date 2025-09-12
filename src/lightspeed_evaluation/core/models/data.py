"""Core data models for evaluation framework."""

import logging
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..constants import SUPPORTED_RESULT_STATUSES

logger = logging.getLogger(__name__)


class TurnData(BaseModel):
    """Individual turn data within a conversation."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str = Field(..., min_length=1, description="Turn ID (alphanumeric)")
    query: str = Field(..., min_length=1, description="Query")
    attachments: Optional[list[str]] = Field(
        default=None, min_length=1, description="Attachments"
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
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID - populated by API if enabled"
    )

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

    # Metrics to run (None = skip that level of evaluation)
    turn_metrics: Optional[list[str]] = Field(
        default=None, description="Turn-level metrics to evaluate"
    )
    conversation_metrics: Optional[list[str]] = Field(
        default=None, description="Conversation-level metrics to evaluate"
    )

    # Metric-specific configuration (threshold, weights, etc.)
    turn_metrics_metadata: Optional[dict[str, dict[str, Any]]] = Field(
        default=None, description="Turn-level metric configuration"
    )
    conversation_metrics_metadata: Optional[dict[str, dict[str, Any]]] = Field(
        default=None, description="Conversation-level metric configuration"
    )

    # Conversation turns
    turns: list[TurnData] = Field(
        ..., min_length=1, description="Conversation turns - must have at least one"
    )

    @field_validator("turn_metrics", "conversation_metrics")
    @classmethod
    def validate_metrics(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Validate metrics are properly formatted."""
        if v is not None:
            for metric in v:
                if not metric or ":" not in metric:
                    raise ValueError(
                        f'Metric "{metric}" must be in format "framework:metric_name"'
                    )
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
