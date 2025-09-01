"""Data models for Evaluation Framework."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class TurnData(BaseModel):
    """Individual turn data within a conversation."""

    turn_id: int
    query: str
    response: str
    contexts: List[Dict[str, str]] = Field(default_factory=list)
    expected_response: str = ""

    @field_validator("query", "response")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        """Validate that query and response are non-empty."""
        if not v or not v.strip():
            raise ValueError("Query and response cannot be empty")
        return v.strip()

    @field_validator("turn_id")
    @classmethod
    def validate_turn_id(cls, v: int) -> int:
        """Validate that turn_id is positive."""
        if v <= 0:
            raise ValueError("Turn ID must be positive")
        return v

    @field_validator("contexts")
    @classmethod
    def validate_contexts(cls, v: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Validate context structure."""
        for i, ctx in enumerate(v):
            if not isinstance(ctx, dict):
                raise ValueError(f"Context {i} must be a dictionary")
            if "content" not in ctx:
                raise ValueError(f'Context {i} must have a "content" field')
            if not ctx["content"].strip():
                raise ValueError(f"Context {i} content cannot be empty")
        return v


class EvaluationData(BaseModel):
    """Complete evaluation data for a conversation group."""

    conversation_group_id: str
    description: str = ""

    # Metrics to run (empty list = skip that level of evaluation)
    turn_metrics: List[str] = Field(default_factory=list)
    conversation_metrics: List[str] = Field(default_factory=list)

    # Metric-specific configuration (threshold, weights, etc.)
    turn_metrics_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    conversation_metrics_metadata: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict
    )

    # Conversation turns
    turns: List[TurnData]

    @field_validator("conversation_group_id")
    @classmethod
    def validate_conversation_group_id(cls, v: str) -> str:
        """Validate conversation group ID is non-empty."""
        if not v or not v.strip():
            raise ValueError("Conversation group ID cannot be empty")
        return v.strip()

    @field_validator("turns")
    @classmethod
    def validate_turns(cls, v: List[TurnData]) -> List[TurnData]:
        """Validate turns list is non-empty."""
        if not v:
            raise ValueError("Conversation must have at least one turn")
        return v

    @field_validator("turn_metrics", "conversation_metrics")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        """Validate metrics are properly formatted."""
        for metric in v:
            if not metric or ":" not in metric:
                raise ValueError(
                    f'Metric "{metric}" must be in format "framework:metric_name"'
                )
        return v

    def validate_metric_requirements(self) -> List[str]:
        """Validate that required fields exist for specified metrics."""
        errors = []

        # Define metric requirements
        context_required_metrics = [
            "ragas:faithfulness",
            "ragas:context_recall",
            "ragas:context_relevance",
            "ragas:context_precision_with_reference",
            "ragas:context_precision_without_reference",
        ]

        expected_response_required_metrics = [
            "ragas:context_recall",
            "ragas:context_precision_with_reference",
            "custom:answer_correctness",
        ]

        # Check each turn against metric requirements
        for turn_data in self.turns:
            # Check context requirements
            for metric in self.turn_metrics:
                if metric in context_required_metrics and not turn_data.contexts:
                    errors.append(
                        f"TurnData {turn_data.turn_id}: Metric '{metric}' requires contexts"
                    )

                if (
                    metric in expected_response_required_metrics
                    and not turn_data.expected_response
                ):
                    errors.append(
                        f"TurnData {turn_data.turn_id}: "
                        f"Metric '{metric}' requires expected_response"
                    )

        return errors


class EvaluationResult(BaseModel):
    """Single evaluation result."""

    conversation_group_id: str
    turn_id: Optional[int]
    metric_identifier: str  # e.g., "ragas:response_relevancy"
    result: str  # PASS/FAIL/ERROR
    score: Optional[float] = None
    threshold: Optional[float] = None
    reason: str = ""
    query: str = ""
    response: str = ""
    execution_time: float = 0.0

    @field_validator("result")
    @classmethod
    def validate_result(cls, v: str) -> str:
        """Validate result is valid status."""
        if v not in ["PASS", "FAIL", "ERROR"]:
            raise ValueError("Result must be PASS, FAIL, or ERROR")
        return v

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate score is between 0 and 1 if present."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Score must be between 0 and 1")
        return v

    @field_validator("execution_time")
    @classmethod
    def validate_execution_time(cls, v: float) -> float:
        """Validate execution time is non-negative."""
        if v < 0:
            raise ValueError("Execution time must be non-negative")
        return v


class LLMConfig(BaseModel):
    """LLM configuration from system configuration."""

    provider: str = Field(
        ..., description="Provider name, e.g., openai, azure, watsonx"
    )
    model: str = Field(..., description="Model identifier or deployment name")
    temperature: float = Field(0.0, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(512, ge=1, description="Maximum tokens in response")
    timeout: int = Field(300, ge=1, description="Request timeout in seconds")
    num_retries: int = Field(3, ge=0, description="Retry attempts for failed requests")

    @field_validator("provider", "model")
    @classmethod
    def _validate_non_empty(cls, v: str) -> str:
        """Validate provider and model are non-empty strings."""
        if not v or not isinstance(v, str) or not v.strip():
            raise ValueError("Value cannot be empty")
        return v.strip()

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LLMConfig":
        """Create LLMConfig from a plain dictionary."""
        return cls(
            provider=config_dict.get("provider", "openai"),
            model=config_dict.get("model", "gpt-4o-mini"),
            temperature=config_dict.get("temperature", 0.0),
            max_tokens=config_dict.get("max_tokens", 512),
            timeout=config_dict.get("timeout", 300),
            num_retries=config_dict.get("num_retries", 3),
        )


# System configuration models
class EvaluationSystemConfig(BaseModel):
    """System configuration for evaluation."""

    llm: Dict[str, Any]
    logging: Dict[str, Any] = Field(default_factory=dict)
    metrics_metadata: Dict[str, Any] = Field(default_factory=dict)
    output: Dict[str, Any] = Field(default_factory=dict)

    default_turn_metrics_metadata: Optional[Dict[str, Any]] = None
    default_conversation_metrics_metadata: Optional[Dict[str, Any]] = None


class OutputConfig(BaseModel):
    """Output configuration."""

    output_dir: str = "./eval_output"
    base_filename: str = "evaluation"
    csv_format: bool = True
    json_format: bool = True
    txt_format: bool = True
    include_graphs: bool = True
    csv_columns: List[str] = Field(default_factory=list)
    visualization_figsize: List[int] = Field(default=[12, 8])
    visualization_dpi: int = 300
    default_turn_metrics_metadata: Optional[Dict[str, Any]] = None
    default_conversation_metrics_metadata: Optional[Dict[str, Any]] = None
