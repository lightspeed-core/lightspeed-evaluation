"""Unit tests for core.config.models module."""

import pytest
from lightspeed_evaluation.core.models import (
    CoreConfig,
    EvaluationData,
    EvaluationResult,
    LLMConfig,
    OutputConfig,
    SystemConfig,
    TurnData,
)
from pydantic import ValidationError


class TestTurnData:
    """Unit tests for TurnData model."""

    def test_valid_turn_data_creation(self):
        """Test creating valid TurnData instance."""
        turn = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Python context"],
            expected_response="Python is a high-level language.",
        )

        assert turn.turn_id == "1"
        assert turn.query == "What is Python?"
        assert turn.response == "Python is a programming language."
        assert turn.contexts is not None
        assert len(turn.contexts) == 1
        assert turn.contexts[0] == "Python context"
        assert turn.expected_response == "Python is a high-level language."

    def test_turn_data_invalid_empty_query(self):
        """Test validation error for empty query."""
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            TurnData(turn_id="1", query="", response="Valid response")

    def test_turn_data_invalid_context_missing_content(self):
        """Test validation error for non-string context."""
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            TurnData(
                turn_id="1",
                query="Valid query",
                response="Valid response",
                contexts=[{"title": "No content field"}],
            )

    def test_turn_data_multiple_contexts(self):
        """Test TurnData with multiple valid contexts."""
        contexts = [
            "First context",
            "Second context",
            "Third context",
        ]
        turn = TurnData(
            turn_id="1", query="Test query", response="Test response", contexts=contexts
        )

        assert len(turn.contexts) == 3
        assert turn.contexts[0] == "First context"
        assert turn.contexts[1] == "Second context"
        assert turn.contexts[2] == "Third context"


class TestEvaluationData:
    """Unit tests for EvaluationData model."""

    def test_valid_evaluation_data_creation(self):
        """Test creating valid EvaluationData instance."""
        turn = TurnData(
            turn_id="1",
            query="Test query",
            response="Test response",
            turn_metrics=["ragas:faithfulness"],
        )
        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            description="Test conversation",
            conversation_metrics=["deepeval:completeness"],
            turns=[turn],
        )

        assert eval_data.conversation_group_id == "test_conv"
        assert eval_data.description == "Test conversation"
        assert eval_data.conversation_metrics == ["deepeval:completeness"]
        assert len(eval_data.turns) == 1
        assert eval_data.turns[0].turn_metrics == ["ragas:faithfulness"]

    def test_evaluation_data_with_minimal_fields(self):
        """Test EvaluationData with only required fields."""
        turn = TurnData(turn_id="1", query="Test query", response="Test response")
        eval_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        assert eval_data.conversation_group_id == "test_conv"
        assert eval_data.description is None
        assert eval_data.conversation_metrics is None
        assert len(eval_data.turns) == 1
        assert eval_data.turns[0].turn_metrics is None

    def test_evaluation_data_invalid_empty_conversation_id(self):
        """Test validation error for empty conversation_group_id."""
        turn = TurnData(turn_id="1", query="Test query", response="Test response")
        with pytest.raises(
            ValidationError, match="String should have at least 1 character"
        ):
            EvaluationData(conversation_group_id="", turns=[turn])

    def test_evaluation_data_invalid_metric_format_missing_colon(self):
        """Test validation error for metric without colon."""
        with pytest.raises(
            ValidationError, match='must be in format "framework:metric_name"'
        ):
            TurnData(
                turn_id="1",
                query="Test query",
                response="Test response",
                turn_metrics=["invalid_metric"],
            )

    def test_evaluation_data_with_metadata(self):
        """Test EvaluationData with metadata fields."""
        turn = TurnData(
            turn_id="1",
            query="Test query",
            response="Test response",
            turn_metrics=["ragas:faithfulness"],
            turn_metrics_metadata={"ragas:faithfulness": {"threshold": 0.8}},
        )
        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            conversation_metrics=["deepeval:completeness"],
            conversation_metrics_metadata={"deepeval:completeness": {"threshold": 0.9}},
            turns=[turn],
        )

        assert eval_data.turns[0].turn_metrics_metadata == {
            "ragas:faithfulness": {"threshold": 0.8}
        }
        assert eval_data.conversation_metrics_metadata == {
            "deepeval:completeness": {"threshold": 0.9}
        }


class TestLLMConfig:
    """Unit tests for LLMConfig model."""

    def test_valid_llm_config_creation(self):
        """Test creating valid LLMConfig instance."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            num_retries=3,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.timeout == 60
        assert config.num_retries == 3

    def test_llm_config_with_defaults(self):
        """Test LLMConfig with default values."""
        config = LLMConfig(provider="openai", model="gpt-4")

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.0
        assert config.max_tokens == 512
        assert config.timeout == 300
        assert config.num_retries == 3


class TestSystemConfig:
    """Unit tests for SystemConfig model."""

    def test_valid_system_config_creation(self):
        """Test creating valid SystemConfig instance."""
        config = SystemConfig(
            core=CoreConfig(max_threads=42),
            llm=LLMConfig(
                provider="anthropic", model="claude-3-sonnet", temperature=0.5
            ),
            output=OutputConfig(output_dir="./custom_output", enabled_outputs=["json"]),
        )

        assert config.llm.provider == "anthropic"
        assert config.llm.model == "claude-3-sonnet"
        assert config.llm.temperature == 0.5
        assert config.output.output_dir == "./custom_output"
        assert config.output.enabled_outputs == ["json"]
        assert config.core.max_threads == 42

    def test_system_config_with_defaults(self):
        """Test SystemConfig with default values."""
        config = SystemConfig()

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o-mini"
        assert config.llm.temperature == 0.0
        assert config.output.output_dir == "./eval_output"
        assert "csv" in config.output.enabled_outputs
        assert config.core.max_threads is None

    def test_system_config_logging_defaults(self):
        """Test SystemConfig logging configuration defaults."""
        config = SystemConfig()

        assert config.logging.source_level == "INFO"
        assert config.logging.package_level == "WARNING"
        assert config.logging.show_timestamps is True
        assert isinstance(config.logging.package_overrides, dict)


class TestEvaluationResult:
    """Unit tests for EvaluationResult model."""

    def test_valid_evaluation_result_creation(self):
        """Test creating valid EvaluationResult instance."""
        result = EvaluationResult(
            conversation_group_id="test_conv",
            turn_id="1",
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.85,
            reason="High faithfulness score",
        )

        assert result.conversation_group_id == "test_conv"
        assert result.turn_id == "1"
        assert result.metric_identifier == "ragas:faithfulness"
        assert result.result == "PASS"
        assert result.score == 0.85
        assert result.reason == "High faithfulness score"

    def test_evaluation_result_conversation_level(self):
        """Test EvaluationResult for conversation-level metric."""
        result = EvaluationResult(
            conversation_group_id="test_conv",
            turn_id=None,
            metric_identifier="deepeval:conversation_completeness",
            result="PASS",
            score=0.92,
            reason="Complete conversation flow",
        )

        assert result.turn_id is None
        assert result.metric_identifier == "deepeval:conversation_completeness"
        assert result.score == 0.92

    def test_evaluation_result_validation_invalid_result(self):
        """Test EvaluationResult validation with invalid result."""
        with pytest.raises(ValidationError, match="Result must be one of"):
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="1",
                metric_identifier="test:metric",
                result="INVALID",
                score=0.5,
            )

    def test_evaluation_result_validation_invalid_score(self):
        """Test EvaluationResult validation with invalid score."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id="1",
                metric_identifier="test:metric",
                result="PASS",
                score=1.5,  # Invalid: > 1
            )
