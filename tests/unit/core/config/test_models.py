"""Unit tests for core.config.models module."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.config.models import (
    EvaluationData,
    EvaluationResult,
    LLMConfig,
    TurnData,
)
from lightspeed_evaluation.core.config.loader import SystemConfig


class TestTurnData:
    """Unit tests for TurnData model."""

    def test_valid_turn_data_creation(self):
        """Test creating valid TurnData instance."""
        turn = TurnData(
            turn_id=1,
            query="What is Python?",
            response="Python is a programming language.",
            contexts=[{"content": "Python context"}],
            expected_response="Python is a high-level language.",
        )

        assert turn.turn_id == 1
        assert turn.query == "What is Python?"
        assert turn.response == "Python is a programming language."
        assert len(turn.contexts) == 1
        assert turn.contexts[0]["content"] == "Python context"
        assert turn.expected_response == "Python is a high-level language."

    def test_turn_data_with_minimal_fields(self):
        """Test TurnData with only required fields."""
        turn = TurnData(turn_id=1, query="Test query", response="Test response")

        assert turn.turn_id == 1
        assert turn.query == "Test query"
        assert turn.response == "Test response"
        assert turn.contexts == []
        assert turn.expected_response == ""

    def test_turn_data_strips_whitespace(self):
        """Test that query and response are stripped of whitespace."""
        turn = TurnData(turn_id=1, query="  Test query  ", response="  Test response  ")

        assert turn.query == "Test query"
        assert turn.response == "Test response"

    def test_turn_data_invalid_empty_fields(self):
        """Test validation error for empty or whitespace-only query/response."""
        with pytest.raises(ValidationError, match="Query and response cannot be empty"):
            TurnData(turn_id=1, query="", response="Valid response")
        
        with pytest.raises(ValidationError, match="Query and response cannot be empty"):
            TurnData(turn_id=1, query="Valid query", response="")
            
        with pytest.raises(ValidationError, match="Query and response cannot be empty"):
            TurnData(turn_id=1, query="   ", response="Valid response")

    def test_turn_data_invalid_turn_id_negative(self):
        """Test validation error for non-positive turn_id."""
        with pytest.raises(ValidationError, match="Turn ID must be positive"):
            TurnData(turn_id=-1, query="Valid query", response="Valid response")

    def test_turn_data_invalid_context_missing_content(self):
        """Test validation error for context missing content field."""
        with pytest.raises(
            ValidationError, match='Context 0 must have a "content" field'
        ):
            TurnData(
                turn_id=1,
                query="Valid query",
                response="Valid response",
                contexts=[{"title": "No content field"}],
            )

    def test_turn_data_invalid_context_empty_content(self):
        """Test validation error for context with empty content."""
        with pytest.raises(ValidationError, match="Context 0 content cannot be empty"):
            TurnData(
                turn_id=1,
                query="Valid query",
                response="Valid response",
                contexts=[{"content": ""}],
            )

    def test_turn_data_multiple_contexts(self):
        """Test TurnData with multiple valid contexts."""
        contexts = [
            {"content": "First context"},
            {"content": "Second context"},
            {"content": "Third context"},
        ]
        turn = TurnData(
            turn_id=1, query="Test query", response="Test response", contexts=contexts
        )

        assert len(turn.contexts) == 3
        assert turn.contexts[0]["content"] == "First context"
        assert turn.contexts[1]["content"] == "Second context"
        assert turn.contexts[2]["content"] == "Third context"


class TestEvaluationData:
    """Unit tests for EvaluationData model."""

    def test_valid_evaluation_data_creation(self):
        """Test creating valid EvaluationData instance."""
        turn = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            description="Test conversation",
            turn_metrics=["ragas:faithfulness"],
            conversation_metrics=["deepeval:completeness"],
            turns=[turn],
        )

        assert eval_data.conversation_group_id == "test_conv"
        assert eval_data.description == "Test conversation"
        assert eval_data.turn_metrics == ["ragas:faithfulness"]
        assert eval_data.conversation_metrics == ["deepeval:completeness"]
        assert len(eval_data.turns) == 1

    def test_evaluation_data_with_minimal_fields(self):
        """Test EvaluationData with only required fields."""
        turn = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        assert eval_data.conversation_group_id == "test_conv"
        assert eval_data.description == ""
        assert eval_data.turn_metrics == []
        assert eval_data.conversation_metrics == []
        assert len(eval_data.turns) == 1

    def test_evaluation_data_invalid_empty_conversation_id(self):
        """Test validation error for empty conversation_group_id."""
        turn = TurnData(turn_id=1, query="Test query", response="Test response")
        with pytest.raises(
            ValidationError, match="Conversation group ID cannot be empty"
        ):
            EvaluationData(conversation_group_id="", turns=[turn])

    def test_evaluation_data_invalid_empty_turns(self):
        """Test validation error for empty turns list."""
        with pytest.raises(
            ValidationError, match="Conversation must have at least one turn"
        ):
            EvaluationData(conversation_group_id="test_conv", turns=[])

    def test_evaluation_data_invalid_metric_format_missing_colon(self):
        """Test validation error for metric without colon."""
        turn = TurnData(turn_id=1, query="Test query", response="Test response")
        with pytest.raises(
            ValidationError, match='must be in format "framework:metric_name"'
        ):
            EvaluationData(
                conversation_group_id="test_conv",
                turn_metrics=["invalid_metric"],
                turns=[turn],
            )

    def test_evaluation_data_multiple_turns(self):
        """Test EvaluationData with multiple turns."""
        turns = [
            TurnData(turn_id=1, query="First query", response="First response"),
            TurnData(turn_id=2, query="Second query", response="Second response"),
            TurnData(turn_id=3, query="Third query", response="Third response"),
        ]
        eval_data = EvaluationData(conversation_group_id="test_conv", turns=turns)

        assert len(eval_data.turns) == 3
        assert eval_data.turns[0].turn_id == 1
        assert eval_data.turns[1].turn_id == 2
        assert eval_data.turns[2].turn_id == 3

    def test_evaluation_data_with_metadata(self):
        """Test EvaluationData with metadata fields."""
        turn = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=["ragas:faithfulness"],
            conversation_metrics=["deepeval:completeness"],
            turn_metrics_metadata={"ragas:faithfulness": {"threshold": 0.8}},
            conversation_metrics_metadata={"deepeval:completeness": {"threshold": 0.9}},
            turns=[turn],
        )

        assert eval_data.turn_metrics_metadata == {
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

    def test_llm_config_from_dict(self):
        """Test creating LLMConfig from dictionary."""
        config_dict = {
            "provider": "anthropic",
            "model": "claude-3-sonnet",
            "temperature": 0.5,
            "max_tokens": 2000,
            "timeout": 90,
            "num_retries": 2,
        }

        config = LLMConfig.from_dict(config_dict)

        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.timeout == 90
        assert config.num_retries == 2

    def test_llm_config_from_dict_with_missing_fields(self):
        """Test creating LLMConfig from dictionary with missing optional fields."""
        config_dict = {"provider": "openai", "model": "gpt-3.5-turbo"}

        config = LLMConfig.from_dict(config_dict)

        assert config.provider == "openai"
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.0  # default
        assert config.max_tokens == 512  # default


class TestSystemConfig:
    """Unit tests for SystemConfig model."""

    def test_valid_system_config_creation(self):
        """Test creating valid SystemConfig instance."""
        config = SystemConfig(
            llm_provider="anthropic",
            llm_model="claude-3-sonnet",
            llm_temperature=0.5,
            output_dir="./custom_output",
            include_graphs=False,
        )

        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-sonnet"
        assert config.llm_temperature == 0.5
        assert config.output_dir == "./custom_output"
        assert config.include_graphs is False

    def test_system_config_with_defaults(self):
        """Test SystemConfig with default values."""
        config = SystemConfig()

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4o-mini"
        assert config.llm_temperature == 0.0
        assert config.output_dir == "./eval_output"
        assert config.include_graphs is True

    def test_system_config_logging_defaults(self):
        """Test SystemConfig logging configuration defaults."""
        config = SystemConfig()

        assert config.logging_source_level == "INFO"
        assert config.logging_package_level == "WARNING"
        assert config.logging_show_timestamps is True
        assert isinstance(config.logging_package_overrides, dict)

    def test_system_config_visualization_defaults(self):
        """Test SystemConfig visualization configuration defaults."""
        config = SystemConfig()

        assert config.visualization_figsize == [12, 8]
        assert config.visualization_dpi == 300

    def test_system_config_custom_values(self):
        """Test SystemConfig with all custom values."""
        config = SystemConfig(
            llm_provider="openai",
            llm_model="gpt-4",
            llm_temperature=0.7,
            llm_max_tokens=1000,
            llm_timeout=60,
            llm_retries=2,
            logging_source_level="DEBUG",
            logging_package_level="ERROR",
            output_dir="./test_output",
            base_filename="test_eval",
            csv_format=False,
            json_format=True,
            txt_format=False,
            include_graphs=True,
            visualization_figsize=[10, 6],
            visualization_dpi=150,
        )

        assert config.llm_provider == "openai"
        assert config.llm_model == "gpt-4"
        assert config.llm_temperature == 0.7
        assert config.llm_max_tokens == 1000
        assert config.llm_timeout == 60
        assert config.llm_retries == 2
        assert config.logging_source_level == "DEBUG"
        assert config.logging_package_level == "ERROR"
        assert config.output_dir == "./test_output"
        assert config.base_filename == "test_eval"
        assert config.csv_format is False
        assert config.json_format is True
        assert config.txt_format is False
        assert config.include_graphs is True
        assert config.visualization_figsize == [10, 6]
        assert config.visualization_dpi == 150


class TestEvaluationResult:
    """Unit tests for EvaluationResult model."""

    def test_valid_evaluation_result_creation(self):
        """Test creating valid EvaluationResult instance."""
        result = EvaluationResult(
            conversation_group_id="test_conv",
            turn_id=1,
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.85,
            reason="High faithfulness score",
        )

        assert result.conversation_group_id == "test_conv"
        assert result.turn_id == 1
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
        with pytest.raises(ValidationError, match="Result must be PASS, FAIL, or ERROR"):
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id=1,
                metric_identifier="test:metric",
                result="INVALID",
                score=0.5,
            )

    def test_evaluation_result_validation_invalid_score(self):
        """Test EvaluationResult validation with invalid score."""
        with pytest.raises(ValidationError, match="Score must be between 0 and 1"):
            EvaluationResult(
                conversation_group_id="test_conv",
                turn_id=1,
                metric_identifier="test:metric",
                result="PASS",
                score=1.5,  # Invalid: > 1
            )