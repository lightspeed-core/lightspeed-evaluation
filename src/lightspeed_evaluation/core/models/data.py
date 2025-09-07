"""Core data models for evaluation framework."""

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..constants import SUPPORTED_RESULT_STATUSES

logger = logging.getLogger(__name__)


class ToolCallData(BaseModel):
    """Individual tool call structure."""

    name: str = Field(..., min_length=1, description="Tool name")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Tool arguments"
    )


class TurnData(BaseModel):
    """Individual turn data within a conversation."""

    turn_id: str = Field(..., min_length=1, description="Turn ID (alphanumeric)")
    query: str = Field(..., min_length=1, description="Query")
    attachments: Optional[List[str]] = Field(
        default=None, min_length=1, description="Attachments"
    )
    response: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Actual Response - populated by API if enabled",
    )
    tool_calls: Optional[List[List[ToolCallData]]] = Field(
        default=None, description="Actual Tool calls - populated by API if enabled"
    )
    contexts: Optional[List[str]] = Field(
        default=None, min_length=1, description="Contexts"
    )
    expected_response: Optional[str] = Field(
        default=None, min_length=1, description="Expected response for comparison"
    )
    expected_tool_calls: Optional[List[List[ToolCallData]]] = Field(
        default=None, description="Expected tool call sequences"
    )
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID - populated by API if enabled"
    )

    @field_validator("expected_tool_calls", mode="before")
    @classmethod
    def validate_expected_tool_calls(
        cls, v: Optional[Any]
    ) -> Optional[List[List[ToolCallData]]]:
        """Validate expected tool calls when provided."""
        if v is None:
            return None

        # If it's already a List[List[ToolCallData]], return it
        if isinstance(v, list) and all(isinstance(seq, list) for seq in v):
            # Check if it's already ToolCallData objects
            if all(isinstance(tool, ToolCallData) for seq in v for tool in seq):
                return v

        # If it's a list format from YAML, convert it to ToolCallData objects
        if isinstance(v, list):
            result = []
            for sequence in v:
                if not isinstance(sequence, list):
                    raise ValueError("Each tool call sequence must be a list")

                tool_calls = []
                for tool_call_dict in sequence:
                    if isinstance(tool_call_dict, ToolCallData):
                        tool_calls.append(tool_call_dict)
                    elif isinstance(tool_call_dict, dict):
                        name = tool_call_dict.get("name")
                        if not name:
                            raise ValueError("Tool call must have 'name' field")

                        tool_calls.append(
                            ToolCallData(
                                name=name, arguments=tool_call_dict.get("arguments", {})
                            )
                        )
                    else:
                        raise ValueError(
                            "Each tool call must be a dictionary or ToolCallData object"
                        )

                result.append(tool_calls)
            return result

        raise ValueError("Expected tool calls must be a list of sequences")


class EvaluationData(BaseModel):
    """Complete evaluation data for a conversation group."""

    conversation_group_id: str = Field(
        ..., min_length=1, description="Unique conversation group identifier"
    )
    description: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional description of the conversation group",
    )

    # Metrics to run (None = skip that level of evaluation)
    turn_metrics: Optional[List[str]] = Field(
        default=None, description="Turn-level metrics to evaluate"
    )
    conversation_metrics: Optional[List[str]] = Field(
        default=None, description="Conversation-level metrics to evaluate"
    )

    # Metric-specific configuration (threshold, weights, etc.)
    turn_metrics_metadata: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None, description="Turn-level metric configuration"
    )
    conversation_metrics_metadata: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None, description="Conversation-level metric configuration"
    )

    # Conversation turns
    turns: List[TurnData] = Field(
        ..., min_length=1, description="Conversation turns - must have at least one"
    )

    @field_validator("turn_metrics", "conversation_metrics")
    @classmethod
    def validate_metrics(cls, v: Optional[List[str]]) -> Optional[List[str]]:
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
