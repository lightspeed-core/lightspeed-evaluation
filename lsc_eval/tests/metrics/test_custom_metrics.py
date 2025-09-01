"""Test cases for Custom Metrics based on system.yaml configuration."""

from typing import Any, Dict, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock

import pytest

from lsc_eval.metrics.custom_metrics import CustomMetrics, EvaluationPromptParams
from lsc_eval.llm_managers.llm_manager import LLMManager, LLMConfig
from lsc_eval.core.models import TurnData
from lsc_eval.output.utils import EvaluationScope


class TestEvaluationPromptParams:
    """Test EvaluationPromptParams Pydantic model."""

    def test_evaluation_prompt_params_creation(self):
        """Test creating EvaluationPromptParams with all fields."""
        params = EvaluationPromptParams(
            metric_name="answer_correctness",
            query="What is machine learning?",
            response="ML is a subset of AI.",
            expected_response="Machine learning is a method of data analysis.",
            contexts=[{"content": "Context about ML"}],
            scale="0.0 to 1.0"
        )

        assert params.metric_name == "answer_correctness"
        assert params.query == "What is machine learning?"
        assert params.response == "ML is a subset of AI."
        assert params.expected_response == "Machine learning is a method of data analysis."
        assert params.contexts == [{"content": "Context about ML"}]
        assert params.scale == "0.0 to 1.0"

    def test_evaluation_prompt_params_minimal(self):
        """Test creating EvaluationPromptParams with minimal required fields."""
        params = EvaluationPromptParams(
            metric_name="answer_correctness",
            query="Test query",
            response="Test response"
        )

        assert params.metric_name == "answer_correctness"
        assert params.query == "Test query"
        assert params.response == "Test response"
        assert params.expected_response is None
        assert params.contexts is None
        assert params.scale == "0.0 to 1.0"  # Default value

    def test_evaluation_prompt_params_custom_scale(self):
        """Test EvaluationPromptParams with custom scale."""
        params = EvaluationPromptParams(
            metric_name="test_metric",
            query="Test query",
            response="Test response",
            scale="1 to 10"
        )

        assert params.scale == "1 to 10"


class TestCustomMetrics:
    """Test CustomMetrics class functionality."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock LLM manager."""
        mock_manager = Mock(spec=LLMManager)
        mock_manager.get_model_name.return_value = "gpt-4o-mini"
        mock_manager.get_litellm_params.return_value = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": 300,
            "num_retries": 3
        }
        return mock_manager

    @pytest.fixture
    def custom_metrics(self, mock_llm_manager):
        """Create CustomMetrics instance with mock LLM manager."""
        with patch('builtins.print'):  # Suppress print statements
            return CustomMetrics(mock_llm_manager)

    @pytest.fixture
    def sample_turn_data(self):
        """Create sample TurnData for testing."""
        return TurnData(
            turn_id=1,
            query="What is machine learning?",
            response="Machine learning is a subset of artificial intelligence.",
            contexts=[
                {"content": "ML involves algorithms that learn from data."},
                {"content": "AI encompasses various techniques including ML."}
            ],
            expected_response="Machine learning is a method of data analysis."
        )

    @pytest.fixture
    def sample_evaluation_scope(self, sample_turn_data):
        """Create sample EvaluationScope for testing."""
        return EvaluationScope(
            turn_idx=0,
            turn_data=sample_turn_data,
            is_conversation=False
        )

    def test_custom_metrics_initialization(self, mock_llm_manager):
        """Test CustomMetrics initialization."""
        with patch('builtins.print') as mock_print:
            metrics = CustomMetrics(mock_llm_manager)

        assert metrics.model_name == "gpt-4o-mini"
        assert metrics.litellm_params["model"] == "gpt-4o-mini"
        assert metrics.litellm_params["temperature"] == 0.0
        assert "answer_correctness" in metrics.supported_metrics
        
        mock_print.assert_called_with("✅ Custom Metrics initialized: gpt-4o-mini")

    def test_evaluate_unsupported_metric(self, custom_metrics, sample_evaluation_scope):
        """Test evaluating an unsupported metric."""
        score, reason = custom_metrics.evaluate(
            "unsupported_metric",
            None,
            sample_evaluation_scope
        )

        assert score is None
        assert "Unsupported custom metric: unsupported_metric" in reason

    @patch('lsc_eval.metrics.custom_metrics.litellm.completion')
    def test_call_llm_success(self, mock_completion, custom_metrics):
        """Test successful LLM call."""
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response content"
        mock_completion.return_value = mock_response

        result = custom_metrics._call_llm("Test prompt")

        assert result == "Test response content"
        mock_completion.assert_called_once()

    @patch('lsc_eval.metrics.custom_metrics.litellm.completion')
    def test_call_llm_with_system_prompt(self, mock_completion, custom_metrics):
        """Test LLM call with system prompt."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Response with system prompt"
        mock_completion.return_value = mock_response

        result = custom_metrics._call_llm("User prompt", "System prompt")

        assert result == "Response with system prompt"
        
        # Check that messages were constructed correctly
        call_args = mock_completion.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == 'System prompt'
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == 'User prompt'

    @patch('lsc_eval.metrics.custom_metrics.litellm.completion')
    def test_call_llm_empty_response(self, mock_completion, custom_metrics):
        """Test LLM call with empty response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = None
        mock_completion.return_value = mock_response

        with pytest.raises(RuntimeError, match="LLM returned empty response"):
            custom_metrics._call_llm("Test prompt")

    @patch('lsc_eval.metrics.custom_metrics.litellm.completion')
    def test_call_llm_exception(self, mock_completion, custom_metrics):
        """Test LLM call with exception."""
        mock_completion.side_effect = Exception("API error")

        with pytest.raises(RuntimeError, match="LiteLLM call failed: API error"):
            custom_metrics._call_llm("Test prompt")

    def test_parse_score_response_explicit_format(self, custom_metrics):
        """Test parsing response with explicit Score/Reason format."""
        response = "Score: 0.85\nReason: The answer is accurate and complete."
        
        score, reason = custom_metrics._parse_score_response(response)
        
        assert score == 0.85
        assert reason == "The answer is accurate and complete."

    def test_parse_score_response_fraction_format(self, custom_metrics):
        """Test parsing response with fraction format."""
        response = "The answer scores 8.5/10 for accuracy."
        
        score, reason = custom_metrics._parse_score_response(response)
        
        assert score == 0.85  # 8.5/10 normalized to 0-1
        assert reason == response
        

    def test_parse_score_response_decimal_only(self, custom_metrics):
        """Test parsing response with decimal number only."""
        response = "The quality is 0.75 based on the criteria."
        
        score, reason = custom_metrics._parse_score_response(response)
        
        assert score == 0.75
        assert reason == response

    def test_parse_score_response_no_score(self, custom_metrics):
        """Test parsing response with no extractable score."""
        response = "The response is good but hard to quantify."
        
        score, reason = custom_metrics._parse_score_response(response)
        
        assert score is None
        assert reason == response

    def test_parse_score_response_scale_normalization(self, custom_metrics):
        """Test score normalization for different scales."""
        test_cases = [
            ("Score: 8.5", 0.85),  # 0-10 scale
            ("Score: 85", 0.85),   # 0-100 scale
            ("Score: 0.85", 0.85), # 0-1 scale (no change)
            ("Score: 150", 150.0), # > 100 scale (no normalization)
        ]

        for response, expected_score in test_cases:
            score, _ = custom_metrics._parse_score_response(response)
            assert score == expected_score

    def test_extract_score_from_text_patterns(self, custom_metrics):
        """Test various score extraction patterns."""
        test_cases = [
            ("8.5/10", 0.85),
            ("4.2/5", 0.84),
            ("3 out of 4", 0.75),
            ("7.5 out of 10", 0.75),
            ("Score is 0.92", 0.92),
            ("Rating: 85", 85.0),
            ("No score here", None),
        ]

        for text, expected_score in test_cases:
            score = custom_metrics._extract_score_from_text(text)
            if expected_score is None:
                assert score is None
            else:
                assert abs(score - expected_score) < 0.01

    def test_create_evaluation_prompt_minimal(self, custom_metrics):
        """Test creating evaluation prompt with minimal parameters."""
        params = EvaluationPromptParams(
            metric_name="test_metric",
            query="Test query",
            response="Test response"
        )

        prompt = custom_metrics._create_evaluation_prompt(params)

        assert "test_metric" in prompt
        assert "Test query" in prompt
        assert "Test response" in prompt
        assert "Score:" in prompt
        assert "Reason:" in prompt

    def test_create_evaluation_prompt_with_expected_response(self, custom_metrics):
        """Test creating evaluation prompt with expected response."""
        params = EvaluationPromptParams(
            metric_name="answer_correctness",
            query="What is ML?",
            response="ML is AI subset",
            expected_response="Machine learning is a subset of AI"
        )

        prompt = custom_metrics._create_evaluation_prompt(params)

        assert "Expected Response: Machine learning is a subset of AI" in prompt

    def test_create_evaluation_prompt_with_contexts(self, custom_metrics):
        """Test creating evaluation prompt with contexts."""
        params = EvaluationPromptParams(
            metric_name="faithfulness",
            query="Test query",
            response="Test response",
            contexts=[
                {"content": "Context 1"},
                {"content": "Context 2"},
                "String context"  # Test mixed context types
            ]
        )

        prompt = custom_metrics._create_evaluation_prompt(params)

        assert "Context:" in prompt
        assert "1. Context 1" in prompt
        assert "2. Context 2" in prompt
        assert "3. String context" in prompt

    def test_evaluate_answer_correctness_success(self, custom_metrics, sample_turn_data):
        """Test successful answer correctness evaluation."""
        with patch.object(custom_metrics, '_call_llm') as mock_call:
            mock_call.return_value = "Score: 0.85\nReason: Accurate and complete answer."
            
            score, reason = custom_metrics._evaluate_answer_correctness(
                None, 0, sample_turn_data, False
            )

        assert score == 0.85
        assert "Custom answer correctness: 0.85" in reason
        assert "Accurate and complete answer" in reason

    def test_evaluate_answer_correctness_conversation_level(self, custom_metrics):
        """Test answer correctness evaluation at conversation level (should fail)."""
        score, reason = custom_metrics._evaluate_answer_correctness(
            None, None, None, True  # is_conversation=True
        )

        assert score is None
        assert "Answer correctness is a turn-level metric" in reason

    def test_evaluate_answer_correctness_no_turn_data(self, custom_metrics):
        """Test answer correctness evaluation without turn data."""
        score, reason = custom_metrics._evaluate_answer_correctness(
            None, 0, None, False  # turn_data=None
        )

        assert score is None
        assert "TurnData is required for answer correctness evaluation" in reason

    def test_evaluate_answer_correctness_unparseable_response(self, custom_metrics, sample_turn_data):
        """Test answer correctness evaluation with unparseable LLM response."""
        with patch.object(custom_metrics, '_call_llm') as mock_call:
            mock_call.return_value = "This response has no score format."
            
            score, reason = custom_metrics._evaluate_answer_correctness(
                None, 0, sample_turn_data, False
            )

        assert score is None
        assert "Could not parse score from LLM response" in reason

    def test_evaluate_answer_correctness_llm_call_failure(self, custom_metrics, sample_turn_data):
        """Test answer correctness evaluation when LLM call fails."""
        with patch.object(custom_metrics, '_call_llm') as mock_call:
            mock_call.side_effect = RuntimeError("LLM call failed")
            
            with pytest.raises(RuntimeError, match="LLM call failed"):
                custom_metrics._evaluate_answer_correctness(
                    None, 0, sample_turn_data, False
                )

    def test_evaluate_answer_correctness_prompt_content(self, custom_metrics, sample_turn_data):
        """Test that answer correctness prompt contains expected content."""
        with patch.object(custom_metrics, '_call_llm') as mock_call:
            mock_call.return_value = "Score: 0.8\nReason: Good answer."
            
            custom_metrics._evaluate_answer_correctness(
                None, 0, sample_turn_data, False
            )

        # Check the prompt passed to _call_llm
        call_args = mock_call.call_args[0][0]
        assert "answer correctness" in call_args
        assert sample_turn_data.query in call_args
        assert sample_turn_data.response in call_args
        assert sample_turn_data.expected_response in call_args
        assert "Factual accuracy" in call_args
        assert "Completeness of information" in call_args

    @patch('lsc_eval.metrics.custom_metrics.LLMManager.from_system_config')
    def test_from_system_config(self, mock_from_system_config):
        """Test creating CustomMetrics from system configuration."""
        mock_manager = Mock(spec=LLMManager)
        mock_manager.get_model_name.return_value = "gpt-4o-mini"
        mock_manager.get_litellm_params.return_value = {"model": "gpt-4o-mini"}
        mock_from_system_config.return_value = mock_manager

        system_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini"
            }
        }

        with patch('builtins.print'):
            metrics = CustomMetrics.from_system_config(system_config)

        assert isinstance(metrics, CustomMetrics)
        mock_from_system_config.assert_called_once_with(system_config)

    def test_supported_metrics_list(self, custom_metrics):
        """Test that supported metrics are properly defined."""
        assert "answer_correctness" in custom_metrics.supported_metrics
        assert callable(custom_metrics.supported_metrics["answer_correctness"])

    def test_litellm_params_usage(self, custom_metrics):
        """Test that LiteLLM parameters are properly used."""
        expected_params = {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "max_tokens": 512,
            "timeout": 300,
            "num_retries": 3
        }

        assert custom_metrics.litellm_params == expected_params

    def test_model_name_usage(self, custom_metrics):
        """Test that model name is properly stored and used."""
        assert custom_metrics.model_name == "gpt-4o-mini"

    def test_evaluation_with_different_turn_data(self, custom_metrics):
        """Test evaluation with different turn data configurations."""
        # Turn data without expected response
        turn_data_no_expected = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        # Turn data without contexts
        turn_data_no_contexts = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response",
            expected_response="Expected response"
        )

        with patch.object(custom_metrics, '_call_llm') as mock_call:
            mock_call.return_value = "Score: 0.7\nReason: Decent answer."

            # Test with no expected response
            scope1 = EvaluationScope(turn_idx=0, turn_data=turn_data_no_expected, is_conversation=False)
            score1, reason1 = custom_metrics.evaluate("answer_correctness", None, scope1)
            assert score1 == 0.7

            # Test with no contexts
            scope2 = EvaluationScope(turn_idx=0, turn_data=turn_data_no_contexts, is_conversation=False)
            score2, reason2 = custom_metrics.evaluate("answer_correctness", None, scope2)
            assert score2 == 0.7

