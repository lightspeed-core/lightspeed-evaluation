"""Data models for agent evaluation."""

from pathlib import Path
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

VALID_EVAL_RESULTS = ["PASS", "FAIL", "ERROR"]
EVAL_TYPE_REQUIREMENTS = {
    "response_eval:accuracy": (
        "expected_response",
        "requires non-empty 'expected_response'",
    ),
    "response_eval:sub-string": (
        "expected_keywords",
        "requires non-empty 'expected_keywords'",
    ),
    "action_eval": ("eval_verify_script", "requires non-empty 'eval_verify_script'"),
    "tool_eval": ("expected_tool_calls", "requires non-empty 'expected_tool_calls'"),
}


def _validate_eval_type(eval_type: str) -> str:
    """Validate evaluation type."""
    if not isinstance(eval_type, str):
        raise ValueError(f"eval_type must be a string, got {type(eval_type).__name__}")

    eval_type = eval_type.strip()
    if not eval_type:
        raise ValueError("eval_type cannot be empty")

    if eval_type not in EVAL_TYPE_REQUIREMENTS:
        allowed = ", ".join(sorted(EVAL_TYPE_REQUIREMENTS.keys()))
        raise ValueError(f"eval_type must be one of [{allowed}], got '{eval_type}'")

    return eval_type


def _validate_script_path(
    script_file: Optional[Union[str, Path]], script_name: str
) -> Optional[Path]:
    """Validate script path exists and convert to absolute Path."""
    if script_file is not None:
        if isinstance(script_file, str):
            script_file = script_file.strip()
            if not script_file:
                raise ValueError(f"{script_name} cannot be empty string")
            script_file = Path(script_file)

        # Convert to absolute path
        script_path = script_file.resolve()

        # Validate file exists
        if not script_path.exists():
            raise ValueError(f"{script_name} file not found: {script_path}")

        if not script_path.is_file():
            raise ValueError(f"{script_name} is not a file: {script_path}")

        return script_path

    return None


def _calculate_stats_by_category(
    results: list["EvaluationResult"],
    key_extractor: Callable[["EvaluationResult"], str],
) -> dict[str, dict[str, Union[int, float]]]:
    """Calculate statistics grouped by a category from each result."""
    category_stats: dict[str, dict[str, Union[int, float]]] = {}

    for result in results:
        category = key_extractor(result)
        if category not in category_stats:
            category_stats[category] = {"passed": 0, "failed": 0, "errored": 0}

        if result.result == "PASS":
            category_stats[category]["passed"] += 1
        elif result.result == "FAIL":
            category_stats[category]["failed"] += 1
        elif result.result == "ERROR":
            category_stats[category]["errored"] += 1

    # Calculate success rates
    for stats in category_stats.values():
        total = stats["passed"] + stats["failed"] + stats["errored"]
        stats["total"] = total
        stats["success_rate"] = (
            round((stats["passed"] / total) * 100, 2) if total > 0 else 0.0
        )

    return category_stats


class EvaluationDataConfig(BaseModel):
    """Single evaluation data configuration."""

    eval_id: str = Field(..., min_length=1, description="Unique evaluation identifier")
    eval_query: str = Field(..., min_length=1, description="Query to send to the agent")
    eval_types: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "List of evaluation types."
            " -> action_eval, tool_eval, response_eval:sub-string, response_eval:accuracy"
        ),
    )
    expected_response: Optional[str] = Field(
        None, min_length=1, description="Expected response for judge-llm"
    )
    expected_keywords: Optional[list[str]] = Field(
        None, min_length=1, description="List of expected keywords for sub-string"
    )
    expected_tool_calls: Optional[list[list[dict[str, Any]]]] = Field(
        None, min_length=1, description="Expected tool calls for tools evaluation"
    )
    eval_verify_script: Optional[Path] = Field(
        None, description="Script path for script evaluation"
    )
    conversation_group: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = Field(
        None, min_length=1, max_length=500, description="Description of this evaluation"
    )

    @field_validator("eval_types")
    @classmethod
    def validate_eval_types(cls, v: list[str]) -> list[str]:
        """Validate evaluation types."""
        if not v:
            raise ValueError("eval_types cannot be empty")

        # Validate each type and remove duplicates while preserving order
        validated_types = []
        seen = set()
        for eval_type in v:
            validated_type = _validate_eval_type(eval_type)
            if validated_type not in seen:
                validated_types.append(validated_type)
                seen.add(validated_type)

        if not validated_types:
            raise ValueError("No valid eval_types provided")

        return validated_types

    @field_validator("expected_keywords")
    @classmethod
    def validate_keywords(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        """Ensure keywords is a list and validate content."""
        if v is not None:
            if not isinstance(v, list):
                v = [v]
            # Remove empty strings and validate
            v = [keyword.strip() for keyword in v if keyword and keyword.strip()]
            if not v:
                raise ValueError("expected_keywords cannot be empty after filtering")
        return v

    @field_validator("eval_verify_script")
    @classmethod
    def validate_script_path(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Validate verify script path exists and convert to absolute Path."""
        return _validate_script_path(v, "eval_verify_script")

    @model_validator(mode="after")
    def validate_eval_requirements(self) -> "EvaluationDataConfig":
        """Validate eval type specific requirements."""
        # Check requirements for each eval type in the list
        for eval_type in self.eval_types:
            if eval_type in EVAL_TYPE_REQUIREMENTS:
                field_name, error_msg = EVAL_TYPE_REQUIREMENTS[eval_type]
                field_value = getattr(self, field_name)
                if not field_value:
                    raise ValueError(f"eval_types containing '{eval_type}' {error_msg}")

        return self


class ConversationDataConfig(BaseModel):
    """Configuration for a conversation group."""

    conversation_group: str = Field(
        ..., min_length=1, max_length=100, description="Conversation group identifier"
    )
    conversation: list[EvaluationDataConfig] = Field(
        ..., min_length=1, description="List of evaluations in this conversation group"
    )
    description: Optional[str] = Field(
        None,
        min_length=1,
        max_length=500,
        description="Description of this conversation group",
    )
    setup_script: Optional[Path] = Field(
        None, description="Setup script path for conversation group"
    )
    cleanup_script: Optional[Path] = Field(
        None, description="Cleanup script path for conversation group"
    )

    @field_validator("conversation_group")
    @classmethod
    def validate_conversation_group(cls, v: str) -> str:
        """Validate conversation group name."""
        v = v.strip()
        if not v:
            raise ValueError("conversation_group cannot be empty")

        return v

    @field_validator("setup_script", "cleanup_script")
    @classmethod
    def validate_script_path(
        cls, v: Optional[Union[str, Path]], info: ValidationInfo
    ) -> Optional[Path]:
        """Validate script path exists and convert to absolute Path."""
        if info.field_name is None:
            raise ValueError("Set a script name for field validator")

        return _validate_script_path(v, info.field_name)

    @model_validator(mode="after")
    def validate_conversation_data(self) -> "ConversationDataConfig":
        """Validate conversation data consistency."""
        if not self.conversation:
            raise ValueError(
                f"Conversation '{self.conversation_group}' must have at least one evaluation"
            )

        # Set conversation group for all evaluations
        for eval_config in self.conversation:
            eval_config.conversation_group = self.conversation_group

        # Check for duplicate eval_ids within conversation
        eval_ids = [eval_config.eval_id for eval_config in self.conversation]
        duplicates = [eval_id for eval_id in eval_ids if eval_ids.count(eval_id) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate eval_id(s) in conversation '{self.conversation_group}': {duplicates}"
            )

        return self


class EvaluationResult(BaseModel):
    """Result of a single evaluation."""

    eval_id: str = Field(..., min_length=1, description="Evaluation identifier")
    query: str = Field(..., min_length=1, description="Query sent to agent")
    response: str = Field(..., description="Agent response")
    tool_calls: Optional[list[list[dict[str, Any]]]] = Field(
        None, description="Tool calls made by agent (for tool_eval type)"
    )
    eval_type: str = Field(..., description="Type of evaluation performed")
    result: str = Field(..., description="Evaluation result")
    conversation_group: Optional[str] = Field(None, description="Conversation group")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    error: Optional[str] = Field(None, description="Error message if any")

    @field_validator("result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        """Validate result is one of the allowed values."""
        if v not in VALID_EVAL_RESULTS:
            raise ValueError(f"Result must be one of {VALID_EVAL_RESULTS}, got '{v}'")
        return v

    @field_validator("eval_type")
    @classmethod
    def validate_eval_type(cls, v: str) -> str:
        """Validate evaluation type."""
        return _validate_eval_type(v)


class EvaluationStats(BaseModel):
    """Statistics for evaluation runs."""

    total_evaluations: int = Field(..., ge=0, description="Total number of evaluations")
    total_conversations: int = Field(
        ..., ge=0, description="Total number of conversations"
    )
    passed: int = Field(..., ge=0, description="Number of passed evaluations")
    failed: int = Field(..., ge=0, description="Number of failed evaluations")
    errored: int = Field(..., ge=0, description="Number of errored evaluations")
    success_rate: float = Field(
        ..., ge=0.0, le=100.0, description="Success rate percentage"
    )
    by_conversation: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Statistics by conversation"
    )
    by_eval_type: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Statistics by evaluation type"
    )

    @classmethod
    def from_results(cls, results: list[EvaluationResult]) -> "EvaluationStats":
        """Create comprehensive statistics from evaluation results."""
        total = len(results)
        passed = sum(1 for r in results if r.result == "PASS")
        failed = sum(1 for r in results if r.result == "FAIL")
        errored = sum(1 for r in results if r.result == "ERROR")
        success_rate = (passed / total * 100) if total > 0 else 0.0

        # Count unique conversations
        conversations: set[str] = set(
            r.conversation_group for r in results if r.conversation_group
        )

        # Calculate statistics by conversation
        by_conversation = _calculate_stats_by_category(
            results, lambda r: r.conversation_group or "unknown"
        )

        # Calculate statistics by eval_type
        by_eval_type = _calculate_stats_by_category(results, lambda r: r.eval_type)

        return cls(
            total_evaluations=total,
            total_conversations=len(conversations),
            passed=passed,
            failed=failed,
            errored=errored,
            success_rate=success_rate,
            by_conversation=by_conversation,
            by_eval_type=by_eval_type,
        )
