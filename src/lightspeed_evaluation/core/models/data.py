"""Core data models for evaluation framework."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightspeed_evaluation.core.constants import SUPPORTED_RESULT_STATUSES

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
    expected_keywords: Optional[list[list[str]]] = Field(
        default=None,
        description="Expected keywords for keyword evaluation (list of alternatives)",
    )
    expected_response: Optional[str] = Field(
        default=None, min_length=1, description="Expected response for comparison"
    )
    expected_tool_calls: Optional[list[list[list[dict[str, Any]]]]] = Field(
        default=None, description="Expected tool call sequences (with alternatives)"
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

    @field_validator("expected_keywords")
    @classmethod
    def validate_expected_keywords(
        cls, v: Optional[list[list[str]]]
    ) -> Optional[list[list[str]]]:
        """Validate expected keywords when provided."""
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("expected_keywords must be a list of lists")

        # Validate each alternative group
        for i, keyword_group in enumerate(v):
            if not isinstance(keyword_group, list):
                raise ValueError(f"expected_keywords[{i}] must be a list of strings")

            if not keyword_group:
                raise ValueError(f"expected_keywords[{i}] cannot be empty")

            for j, keyword in enumerate(keyword_group):
                if not isinstance(keyword, str):
                    raise ValueError(f"expected_keywords[{i}][{j}] must be a string")
                if not keyword.strip():
                    raise ValueError(
                        f"expected_keywords[{i}][{j}] cannot be empty or whitespace"
                    )

        return v

    @field_validator("expected_tool_calls", mode="before")
    @classmethod
    def validate_expected_tool_calls(
        cls, v: Optional[Any]
    ) -> Optional[list[list[list[dict[str, Any]]]]]:
        """Validate expected tool calls when provided.

        Converts single set format to multiple sets format automatically for backward compatibility.

        Input formats:
        1. Single set: [[{tool_name, arguments}, ...], ...] -> Converted to multiple sets
        2. Multiple sets: [[[{tool_name, arguments}, ...], ...], [[...], ...], ...] -> Used as-is

        Output format:
        Multiple sets: [[[{tool_name, arguments}, ...], ...], [[...], ...], ...]
        """
        if v is None:
            return None

        if not isinstance(v, list):
            raise ValueError("Expected tool calls must be a list")

        # Ensure multiple sets format (backward compatibility)
        data = cls._ensure_multiple_sets_format(v)

        # Validate multiple sets format
        return cls._validate_multiple_sets(data)

        # Future backward compatibility removal (minimal changes):
        # 1. Delete: _ensure_multiple_sets_format() and _is_single_set_format()
        # 2. Replace above with: return cls._validate_multiple_sets(v)

    @classmethod
    def _ensure_multiple_sets_format(cls, v: list) -> list[list[list[dict[str, Any]]]]:
        """Ensure data is in multiple sets format (backward compatibility)."""
        # Convert single set format to multiple sets format if needed
        if cls._is_single_set_format(v):
            # Single set: [[tool1, tool2], [tool3]] -> Multiple sets: [[[tool1, tool2], [tool3]]]
            return [v]
        # Already multiple sets: [[[tool1]], [[tool2]]] -> Keep as-is
        return v

    @classmethod
    def _validate_multiple_sets(
        cls, data: list[list[list[dict[str, Any]]]]
    ) -> list[list[list[dict[str, Any]]]]:
        """Validate multiple sets format data."""
        # Reject empty sequences anywhere
        cls._reject_empty_sequences(data)

        # Validate each alternative set
        validated_alternatives = []
        for alternative in data:
            validated_alternative = cls._validate_tool_call_sequences(alternative)
            validated_alternatives.append(validated_alternative)

        # Apply constraints
        cls._validate_empty_set_constraints(validated_alternatives)
        return validated_alternatives

    @classmethod
    def _is_single_set_format(cls, v: list) -> bool:
        """Detect if input is single set format (backward compatibility)."""
        if not v:
            return True  # Empty list is single set format

        # Check first element: if it's a dict, it's single set format
        # If it's a list, it could be multiple sets or single set with sequences
        first_element = v[0]
        if isinstance(first_element, dict):
            return True  # Single set: [tool1, tool2, ...]

        if isinstance(first_element, list):
            if not first_element:
                # Empty list [] - could be single set (empty sequence) or multiple sets (empty alt)
                # Check if there are multiple empty lists (indicates multiple sets format)
                return not (
                    len(v) > 1
                    and all(isinstance(el, list) and len(el) == 0 for el in v)
                )
            # Non-empty list - check what's inside
            return isinstance(
                first_element[0], dict
            )  # dict = single set, list = multiple sets

        return False

    @classmethod
    def _reject_empty_sequences(cls, data: list[list[list[dict[str, Any]]]]) -> None:
        """Reject empty sequences in data."""
        for i, alternative in enumerate(data):
            for j, sequence in enumerate(alternative):
                if isinstance(sequence, list) and len(sequence) == 0:
                    raise ValueError(
                        f"Empty sequence at position {j} in alternative {i} is invalid. "
                        "Use [] for no tools instead."
                    )

    @classmethod
    def _validate_empty_set_constraints(
        cls, result: list[list[list[dict[str, Any]]]]
    ) -> None:
        """Validate that empty alternatives come after primary options (not first or only)."""
        if not result:
            return

        if len(result) == 1 and len(result[0]) == 0:
            raise ValueError(
                "Empty set cannot be the only alternative. "
                "Empty alternatives should represent fallback scenarios, not primary options."
            )

        if len(result) > 1 and len(result[0]) == 0:
            raise ValueError(
                "Empty set cannot be the first alternative. "
                "Empty alternatives should come after primary options."
            )

        # Prevent multiple redundant empty alternatives
        empty_count = sum(1 for alt in result if len(alt) == 0)
        if empty_count > 1:
            raise ValueError(
                f"Found {empty_count} empty alternatives. "
                "Multiple empty alternatives are redundant - use only one as fallback."
            )

    @classmethod
    def _is_sequence_of_sequences(cls, seq: list) -> bool:
        """Check if a sequence contains sequences (not dicts)."""
        return bool(seq) and isinstance(seq[0], list)

    @classmethod
    def _validate_tool_call_sequences(cls, v: Any) -> list[list[dict[str, Any]]]:
        """Validate tool call sequences structure."""
        if not isinstance(v, list):
            raise ValueError("Expected tool calls must be a list of sequences")

        validated_sequences = []
        for i, sequence in enumerate(v):
            if not isinstance(sequence, list):
                raise ValueError(f"Sequence {i} must be a list")

            # Empty sequences are already rejected by _reject_empty_sequences
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

            validated_sequences.append(tool_calls)
        return validated_sequences


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

    # True when the data fails the validation of metrics requirements/availability
    _invalid_data: bool = False

    # Script execution support
    setup_script: Optional[Union[str, Path]] = Field(
        default=None,
        description="Path to setup script to run before conversation starts",
    )
    cleanup_script: Optional[Union[str, Path]] = Field(
        default=None,
        description="Path to cleanup script to run after conversation ends",
    )

    def mark_data_invalid(self) -> None:
        """Mark the data as invalid."""
        self._invalid_data = True

    def is_data_valid(self) -> bool:
        """Did the data passed the validation checks?"""
        return not self._invalid_data

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
