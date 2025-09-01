"""Test cases for Pydantic models based on system.yaml configuration."""

from typing import Any, Dict, List
import pytest
from pydantic import ValidationError

from lsc_eval.core.models import (
    TurnData,
    EvaluationData,
    EvaluationResult,
    LLMConfig,
    EvaluationSystemConfig,
    OutputConfig,
)


class TestTurnData:
    """Test TurnData Pydantic model."""

    def test_turn_data_valid_creation(self):
        """Test creating valid TurnData instance."""
        turn_data = TurnData(
            turn_id=1,
            query="What is machine learning?",
            response="Machine learning is a subset of AI.",
            contexts=[
                {"content": "ML involves algorithms that learn from data."},
                {"content": "AI encompasses various techniques."}
            ],
            expected_response="ML is a method of data analysis."
        )

        assert turn_data.turn_id == 1
        assert turn_data.query == "What is machine learning?"
        assert turn_data.response == "Machine learning is a subset of AI."
        assert len(turn_data.contexts) == 2
        assert turn_data.expected_response == "ML is a method of data analysis."

    def test_turn_data_minimal_valid(self):
        """Test creating TurnData with minimal required fields."""
        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        assert turn_data.turn_id == 1
        assert turn_data.query == "Test query"
        assert turn_data.response == "Test response"
        assert turn_data.contexts == []
        assert turn_data.expected_response == ""

    def test_turn_data_invalid_turn_id(self):
        """Test TurnData with invalid turn_id."""
        with pytest.raises(ValidationError) as exc_info:
            TurnData(
                turn_id=0,  # Invalid: must be positive
                query="Test query",
                response="Test response"
            )
        
        assert "Turn ID must be positive" in str(exc_info.value)

    def test_turn_data_validation_errors(self):
        """Test TurnData validation errors."""
        # Test empty query
        with pytest.raises(ValidationError):
            TurnData(turn_id=1, query="", response="Test response")
        
        # Test empty response
        with pytest.raises(ValidationError):
            TurnData(turn_id=1, query="Test query", response="")


    def test_turn_data_string_trimming(self):
        """Test that strings are properly trimmed."""
        turn_data = TurnData(
            turn_id=1,
            query="  Test query  ",
            response="  Test response  "
        )

        assert turn_data.query == "Test query"
        assert turn_data.response == "Test response"


class TestEvaluationData:
    """Test EvaluationData Pydantic model."""

    def test_evaluation_data_valid_creation(self):
        """Test creating valid EvaluationData instance."""
        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response",
            contexts=[{"content": "Test context"}],
            expected_response="Expected response"
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_group_1",
            description="Test conversation",
            turn_metrics=["ragas:faithfulness", "ragas:response_relevancy"],
            conversation_metrics=["deepeval:conversation_completeness"],
            turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.8}
            },
            conversation_metrics_metadata={
                "deepeval:conversation_completeness": {"threshold": 0.7}
            },
            turns=[turn_data]
        )

        assert eval_data.conversation_group_id == "conv_group_1"
        assert eval_data.description == "Test conversation"
        assert len(eval_data.turn_metrics) == 2
        assert len(eval_data.conversation_metrics) == 1
        assert len(eval_data.turns) == 1

    def test_evaluation_data_minimal_valid(self):
        """Test creating EvaluationData with minimal required fields."""
        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_group_1",
            turns=[turn_data]
        )

        assert eval_data.conversation_group_id == "conv_group_1"
        assert eval_data.description == ""
        assert eval_data.turn_metrics == []
        assert eval_data.conversation_metrics == []
        assert len(eval_data.turns) == 1

    def test_evaluation_data_invalid_conversation_group_id(self):
        """Test EvaluationData with invalid conversation_group_id."""
        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        with pytest.raises(ValidationError) as exc_info:
            EvaluationData(
                conversation_group_id="",  # Invalid: empty ID
                turns=[turn_data]
            )
        
        assert "Conversation group ID cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            EvaluationData(
                conversation_group_id="   ",  # Invalid: whitespace only
                turns=[turn_data]
            )

    def test_evaluation_data_empty_turns(self):
        """Test EvaluationData with empty turns list."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationData(
                conversation_group_id="conv_group_1",
                turns=[]  # Invalid: must have at least one turn
            )
        
        assert "Conversation must have at least one turn" in str(exc_info.value)

    def test_evaluation_data_invalid_metric_format(self):
        """Test EvaluationData with invalid metric format."""
        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        with pytest.raises(ValidationError) as exc_info:
            EvaluationData(
                conversation_group_id="conv_group_1",
                turn_metrics=["invalid_metric"],  # Invalid: missing colon
                turns=[turn_data]
            )
        
        assert 'Metric "invalid_metric" must be in format "framework:metric_name"' in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            EvaluationData(
                conversation_group_id="conv_group_1",
                conversation_metrics=[""],  # Invalid: empty metric
                turns=[turn_data]
            )

    def test_evaluation_data_validate_metric_requirements(self):
        """Test metric requirements validation."""
        # Test context-required metrics without contexts
        turn_without_contexts = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
            # No contexts provided
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_group_1",
            turn_metrics=["ragas:faithfulness"],  # Requires contexts
            turns=[turn_without_contexts]
        )

        errors = eval_data.validate_metric_requirements()
        assert len(errors) > 0
        assert "Metric 'ragas:faithfulness' requires contexts" in errors[0]

        # Test expected_response-required metrics without expected_response
        turn_without_expected = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response",
            contexts=[{"content": "Context"}]
            # No expected_response provided
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_group_1",
            turn_metrics=["ragas:context_recall"],  # Requires expected_response
            turns=[turn_without_expected]
        )

        errors = eval_data.validate_metric_requirements()
        assert len(errors) > 0
        assert "Metric 'ragas:context_recall' requires expected_response" in errors[0]

    def test_evaluation_data_valid_metric_requirements(self):
        """Test valid metric requirements."""
        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response",
            contexts=[{"content": "Test context"}],
            expected_response="Expected response"
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_group_1",
            turn_metrics=[
                "ragas:faithfulness",  # Requires contexts
                "ragas:context_recall",  # Requires contexts and expected_response
                "custom:answer_correctness"  # Requires expected_response
            ],
            turns=[turn_data]
        )

        errors = eval_data.validate_metric_requirements()
        assert len(errors) == 0

    def test_evaluation_data_multiple_turns(self):
        """Test EvaluationData with multiple turns."""
        turns = [
            TurnData(turn_id=1, query="Query 1", response="Response 1"),
            TurnData(turn_id=2, query="Query 2", response="Response 2"),
            TurnData(turn_id=3, query="Query 3", response="Response 3")
        ]

        eval_data = EvaluationData(
            conversation_group_id="conv_group_1",
            turns=turns
        )

        assert len(eval_data.turns) == 3
        assert eval_data.turns[0].turn_id == 1
        assert eval_data.turns[2].turn_id == 3


class TestEvaluationResult:
    """Test EvaluationResult Pydantic model."""

    def test_evaluation_result_valid_creation(self):
        """Test creating valid EvaluationResult instance."""
        result = EvaluationResult(
            conversation_group_id="conv_group_1",
            turn_id=1,
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.85,
            threshold=0.8,
            reason="Response is faithful to context",
            query="Test query",
            response="Test response",
            execution_time=1.5
        )

        assert result.conversation_group_id == "conv_group_1"
        assert result.turn_id == 1
        assert result.metric_identifier == "ragas:faithfulness"
        assert result.result == "PASS"
        assert result.score == 0.85
        assert result.threshold == 0.8
        assert result.execution_time == 1.5

    def test_evaluation_result_minimal_valid(self):
        """Test creating EvaluationResult with minimal required fields."""
        result = EvaluationResult(
            conversation_group_id="conv_group_1",
            turn_id=None,  # For conversation-level metrics
            metric_identifier="deepeval:conversation_completeness",
            result="FAIL"
        )

        assert result.conversation_group_id == "conv_group_1"
        assert result.turn_id is None
        assert result.result == "FAIL"
        assert result.score is None
        assert result.threshold is None
        assert result.reason == ""
        assert result.execution_time == 0.0

    def test_evaluation_result_invalid_result_status(self):
        """Test EvaluationResult with invalid result status."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                conversation_group_id="conv_group_1",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="INVALID_STATUS"  # Invalid: must be PASS/FAIL/ERROR
            )
        
        assert "Result must be PASS, FAIL, or ERROR" in str(exc_info.value)

    def test_evaluation_result_invalid_score(self):
        """Test EvaluationResult with invalid score values."""
        # Score below 0
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                conversation_group_id="conv_group_1",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=-0.1  # Invalid: below 0
            )
        
        assert "Score must be between 0 and 1" in str(exc_info.value)

        # Score above 1
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                conversation_group_id="conv_group_1",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                score=1.1  # Invalid: above 1
            )

    def test_evaluation_result_invalid_execution_time(self):
        """Test EvaluationResult with invalid execution time."""
        with pytest.raises(ValidationError) as exc_info:
            EvaluationResult(
                conversation_group_id="conv_group_1",
                turn_id=1,
                metric_identifier="ragas:faithfulness",
                result="PASS",
                execution_time=-1.0  # Invalid: negative time
            )
        
        assert "Execution time must be non-negative" in str(exc_info.value)

    def test_evaluation_result_valid_edge_cases(self):
        """Test EvaluationResult with valid edge case values."""
        # Score at boundaries
        result1 = EvaluationResult(
            conversation_group_id="conv_group_1",
            turn_id=1,
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.0  # Valid: at lower boundary
        )
        assert result1.score == 0.0

        result2 = EvaluationResult(
            conversation_group_id="conv_group_1",
            turn_id=1,
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=1.0  # Valid: at upper boundary
        )
        assert result2.score == 1.0

        # Zero execution time
        result3 = EvaluationResult(
            conversation_group_id="conv_group_1",
            turn_id=1,
            metric_identifier="ragas:faithfulness",
            result="PASS",
            execution_time=0.0  # Valid: zero time
        )
        assert result3.execution_time == 0.0


class TestLLMConfig:
    """Test LLMConfig Pydantic model."""

    def test_llm_config_valid_creation(self):
        """Test creating valid LLMConfig instance."""
        config = LLMConfig(
            model_name="gpt-4o-mini",
            api_base="https://api.openai.com/v1",
            api_key="sk-test-key",
            temperature=0.5,
            max_tokens=1024,
            timeout=300,
            num_retries=3
        )

        assert config.model_name == "gpt-4o-mini"
        assert config.api_base == "https://api.openai.com/v1"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.timeout == 300
        assert config.num_retries == 3

    def test_llm_config_minimal_valid(self):
        """Test creating LLMConfig with minimal required fields."""
        config = LLMConfig(model_name="gpt-4o-mini")

        assert config.model_name == "gpt-4o-mini"
        assert config.api_base is None
        assert config.api_key is None
        assert config.temperature == 0.0
        assert config.max_tokens is None
        assert config.timeout is None
        assert config.num_retries == 3

    def test_llm_config_invalid_model_name(self):
        """Test LLMConfig with invalid model name."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(model_name="")  # Invalid: empty model name
        
        assert "Model name cannot be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(model_name="   ")  # Invalid: whitespace only

    def test_llm_config_invalid_temperature(self):
        """Test LLMConfig with invalid temperature values."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                model_name="gpt-4o-mini",
                temperature=-0.1  # Invalid: below 0
            )

        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                model_name="gpt-4o-mini",
                temperature=2.1  # Invalid: above 2
            )

    def test_llm_config_invalid_max_tokens(self):
        """Test LLMConfig with invalid max_tokens."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                model_name="gpt-4o-mini",
                max_tokens=0  # Invalid: must be >= 1
            )

    def test_llm_config_invalid_timeout(self):
        """Test LLMConfig with invalid timeout."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                model_name="gpt-4o-mini",
                timeout=0  # Invalid: must be >= 1
            )

    def test_llm_config_invalid_num_retries(self):
        """Test LLMConfig with invalid num_retries."""
        with pytest.raises(ValidationError) as exc_info:
            LLMConfig(
                model_name="gpt-4o-mini",
                num_retries=-1  # Invalid: must be >= 0
            )

    def test_llm_config_string_trimming(self):
        """Test that model name is properly trimmed."""
        config = LLMConfig(model_name="  gpt-4o-mini  ")
        assert config.model_name == "gpt-4o-mini"

    def test_llm_config_boundary_values(self):
        """Test LLMConfig with boundary values."""
        config = LLMConfig(
            model_name="test-model",
            temperature=0.0,  # Lower boundary
            max_tokens=1,     # Lower boundary
            timeout=1,        # Lower boundary
            num_retries=0     # Lower boundary
        )

        assert config.temperature == 0.0
        assert config.max_tokens == 1
        assert config.timeout == 1
        assert config.num_retries == 0

        config2 = LLMConfig(
            model_name="test-model",
            temperature=2.0   # Upper boundary
        )
        assert config2.temperature == 2.0


class TestEvaluationSystemConfig:
    """Test EvaluationSystemConfig Pydantic model."""

    def test_evaluation_system_config_valid_creation(self):
        """Test creating valid EvaluationSystemConfig instance."""
        config = EvaluationSystemConfig(
            llm={
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0
            },
            logging={
                "source_level": "INFO",
                "package_level": "ERROR"
            },
            metrics_metadata={
                "turn_level": {
                    "ragas:faithfulness": {"threshold": 0.8}
                }
            },
            output={
                "base_directory": "./eval_output",
                "formats": {"csv": True}
            }
        )

        assert config.llm["provider"] == "openai"
        assert config.logging["source_level"] == "INFO"
        assert "ragas:faithfulness" in config.metrics_metadata["turn_level"]
        assert config.output["base_directory"] == "./eval_output"

    def test_evaluation_system_config_minimal_valid(self):
        """Test creating EvaluationSystemConfig with minimal required fields."""
        config = EvaluationSystemConfig(
            llm={"provider": "openai", "model": "gpt-4o-mini"}
        )

        assert config.llm["provider"] == "openai"
        assert config.logging == {}
        assert config.metrics_metadata == {}
        assert config.output == {}

    def test_evaluation_system_config_with_defaults(self):
        """Test EvaluationSystemConfig with default metadata."""
        turn_metadata = {"ragas:faithfulness": {"threshold": 0.8}}
        conversation_metadata = {"deepeval:completeness": {"threshold": 0.7}}

        config = EvaluationSystemConfig(
            llm={"provider": "openai"},
            default_turn_metrics_metadata=turn_metadata,
            default_conversation_metrics_metadata=conversation_metadata
        )

        assert config.default_turn_metrics_metadata == turn_metadata
        assert config.default_conversation_metrics_metadata == conversation_metadata


class TestOutputConfig:
    """Test OutputConfig Pydantic model."""

    def test_output_config_valid_creation(self):
        """Test creating valid OutputConfig instance."""
        config = OutputConfig(
            output_dir="/custom/output",
            base_filename="custom_eval",
            csv_format=True,
            json_format=False,
            txt_format=True,
            include_graphs=False,
            csv_columns=["col1", "col2"],
            visualization_figsize=[10, 6],
            visualization_dpi=150
        )

        assert config.output_dir == "/custom/output"
        assert config.base_filename == "custom_eval"
        assert config.csv_format is True
        assert config.json_format is False
        assert config.include_graphs is False
        assert config.csv_columns == ["col1", "col2"]
        assert config.visualization_figsize == [10, 6]
        assert config.visualization_dpi == 150

    def test_output_config_default_values(self):
        """Test OutputConfig with default values."""
        config = OutputConfig()

        assert config.output_dir == "./eval_output"
        assert config.base_filename == "evaluation"
        assert config.csv_format is True
        assert config.json_format is True
        assert config.txt_format is True
        assert config.include_graphs is True
        assert config.csv_columns == []
        assert config.visualization_figsize == [12, 8]
        assert config.visualization_dpi == 300

    def test_output_config_with_metadata(self):
        """Test OutputConfig with metadata fields."""
        turn_metadata = {"ragas:faithfulness": {"threshold": 0.8}}
        conversation_metadata = {"deepeval:completeness": {"threshold": 0.7}}

        config = OutputConfig(
            default_turn_metrics_metadata=turn_metadata,
            default_conversation_metrics_metadata=conversation_metadata
        )

        assert config.default_turn_metrics_metadata == turn_metadata
        assert config.default_conversation_metrics_metadata == conversation_metadata

