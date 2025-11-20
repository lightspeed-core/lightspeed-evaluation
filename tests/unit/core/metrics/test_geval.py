"""Tests for GEval metrics handler."""

from unittest.mock import MagicMock, patch

import pytest
from deepeval.test_case import LLMTestCaseParams

from lightspeed_evaluation.core.metrics.geval import GEvalHandler
from lightspeed_evaluation.core.metrics.manager import MetricLevel


class TestGEvalHandler:
    """Test cases for GEvalHandler class."""

    @pytest.fixture
    def mock_llm_manager(self):
        """Create a mock DeepEvalLLMManager."""
        mock_manager = MagicMock()
        mock_llm = MagicMock()
        mock_manager.get_llm.return_value = mock_llm
        return mock_manager

    @pytest.fixture
    def mock_metric_manager(self):
        """Create a mock MetricManager."""
        return MagicMock()

    @pytest.fixture
    def handler(self, mock_llm_manager, mock_metric_manager):
        """Create a GEvalHandler instance with mocked dependencies."""
        return GEvalHandler(
            deepeval_llm_manager=mock_llm_manager,
            metric_manager=mock_metric_manager,
        )

    def test_initialization(self, mock_llm_manager, mock_metric_manager):
        """Test GEvalHandler initialization with required dependencies."""
        handler = GEvalHandler(
            deepeval_llm_manager=mock_llm_manager,
            metric_manager=mock_metric_manager,
        )

        assert handler.deepeval_llm_manager == mock_llm_manager
        assert handler.metric_manager == mock_metric_manager

    def test_convert_evaluation_params_field_names(self, handler):
        """Test conversion of evaluation data field names to LLMTestCaseParams enum."""
        params = ["query", "response", "expected_response"]
        result = handler._convert_evaluation_params(params)

        assert result is not None
        assert len(result) == 3
        assert LLMTestCaseParams.INPUT in result
        assert LLMTestCaseParams.ACTUAL_OUTPUT in result
        assert LLMTestCaseParams.EXPECTED_OUTPUT in result

    def test_convert_evaluation_params_with_contexts(self, handler):
        """Test conversion including contexts and retrieval_context fields."""
        params = ["query", "response", "contexts", "retrieval_context"]
        result = handler._convert_evaluation_params(params)

        assert result is not None
        assert len(result) == 4
        assert LLMTestCaseParams.INPUT in result
        assert LLMTestCaseParams.ACTUAL_OUTPUT in result
        assert LLMTestCaseParams.CONTEXT in result
        assert LLMTestCaseParams.RETRIEVAL_CONTEXT in result

    def test_convert_evaluation_params_enum_values_backward_compat(self, handler):
        """Test conversion with direct enum value strings (backward compatibility)."""
        params = ["INPUT", "ACTUAL_OUTPUT", "EXPECTED_OUTPUT"]
        result = handler._convert_evaluation_params(params)

        assert result is not None
        assert len(result) == 3
        assert LLMTestCaseParams.INPUT in result
        assert LLMTestCaseParams.ACTUAL_OUTPUT in result
        assert LLMTestCaseParams.EXPECTED_OUTPUT in result

    def test_convert_evaluation_params_invalid_returns_none(self, handler):
        """Test that invalid params return None to allow GEval auto-detection."""
        params = ["invalid_param", "another_invalid"]
        result = handler._convert_evaluation_params(params)

        assert result is None

    def test_convert_evaluation_params_empty_returns_none(self, handler):
        """Test that empty params list returns None."""
        result = handler._convert_evaluation_params([])

        assert result is None

    def test_convert_evaluation_params_mixed_invalid_returns_none(self, handler):
        """Test that any invalid param causes None return."""
        params = ["query", "invalid_param", "response"]
        result = handler._convert_evaluation_params(params)

        # Should return None because of the invalid param
        assert result is None

    def test_get_geval_config_uses_metric_manager(self, handler, mock_metric_manager):
        """Test that _get_geval_config delegates to MetricManager."""
        expected_config = {
            "criteria": "Test criteria",
            "evaluation_params": ["query", "response"],
            "threshold": 0.8,
        }
        mock_metric_manager.get_metric_metadata.return_value = expected_config

        conv_data = MagicMock()
        config = handler._get_geval_config(
            metric_name="test_metric",
            conv_data=conv_data,
            turn_data=None,
            is_conversation=True,
        )

        assert config == expected_config
        mock_metric_manager.get_metric_metadata.assert_called_once_with(
            metric_identifier="geval:test_metric",
            level=MetricLevel.CONVERSATION,
            conv_data=conv_data,
            turn_data=None,
        )

    def test_get_geval_config_turn_level(self, handler, mock_metric_manager):
        """Test retrieving turn-level config uses correct MetricLevel."""
        expected_config = {"criteria": "Turn criteria", "threshold": 0.9}
        mock_metric_manager.get_metric_metadata.return_value = expected_config

        conv_data = MagicMock()
        turn_data = MagicMock()

        config = handler._get_geval_config(
            metric_name="turn_metric",
            conv_data=conv_data,
            turn_data=turn_data,
            is_conversation=False,
        )

        assert config == expected_config
        mock_metric_manager.get_metric_metadata.assert_called_once_with(
            metric_identifier="geval:turn_metric",
            level=MetricLevel.TURN,
            conv_data=conv_data,
            turn_data=turn_data,
        )

    def test_get_geval_config_returns_none_when_not_found(
        self, handler, mock_metric_manager
    ):
        """Test that None is returned when MetricManager finds no config."""
        mock_metric_manager.get_metric_metadata.return_value = None

        conv_data = MagicMock()
        config = handler._get_geval_config(
            metric_name="nonexistent_metric",
            conv_data=conv_data,
            turn_data=None,
            is_conversation=True,
        )

        assert config is None

    def test_evaluate_missing_config(self, handler, mock_metric_manager):
        """Test that evaluate returns error when config is not found."""
        mock_metric_manager.get_metric_metadata.return_value = None

        conv_data = MagicMock()
        score, reason = handler.evaluate(
            metric_name="nonexistent",
            conv_data=conv_data,
            _turn_idx=0,
            turn_data=None,
            is_conversation=True,
        )

        assert score is None
        assert "configuration not found" in reason.lower()

    def test_evaluate_missing_criteria(self, handler, mock_metric_manager):
        """Test that evaluate requires 'criteria' in config."""
        mock_metric_manager.get_metric_metadata.return_value = {
            "threshold": 0.8,
            "evaluation_params": ["query", "response"],
            # Missing 'criteria'
        }

        conv_data = MagicMock()
        score, reason = handler.evaluate(
            metric_name="test_metric",
            conv_data=conv_data,
            _turn_idx=0,
            turn_data=None,
            is_conversation=True,
        )

        assert score is None
        assert "criteria" in reason.lower()

    def test_evaluate_turn_missing_turn_data(self, handler, mock_metric_manager):
        """Test that turn-level evaluation requires turn_data."""
        mock_metric_manager.get_metric_metadata.return_value = {
            "criteria": "Test criteria"
        }

        conv_data = MagicMock()
        score, reason = handler.evaluate(
            metric_name="test_metric",
            conv_data=conv_data,
            _turn_idx=0,
            turn_data=None,  # Missing required turn data
            is_conversation=False,  # Turn-level
        )

        assert score is None
        assert "turn data required" in reason.lower()

    def test_evaluate_turn_success(self, handler, mock_metric_manager):
        """Test successful turn-level evaluation."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            # Mock GEval metric instance
            mock_metric = MagicMock()
            mock_metric.score = 0.85
            mock_metric.reason = "Test passed"
            mock_geval_class.return_value = mock_metric

            # Setup metric manager to return config
            mock_metric_manager.get_metric_metadata.return_value = {
                "criteria": "Test criteria",
                "evaluation_params": ["query", "response"],
                "evaluation_steps": ["Step 1", "Step 2"],
                "threshold": 0.7,
            }

            # Mock turn data
            turn_data = MagicMock()
            turn_data.query = "Test query"
            turn_data.response = "Test response"
            turn_data.expected_response = None
            turn_data.contexts = None

            conv_data = MagicMock()

            score, reason = handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                _turn_idx=0,
                turn_data=turn_data,
                is_conversation=False,
            )

            assert score == 0.85
            assert reason == "Test passed"
            mock_metric.measure.assert_called_once()

    def test_evaluate_turn_with_optional_fields(self, handler, mock_metric_manager):
        """Test turn-level evaluation includes optional fields when present."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            with patch(
                "lightspeed_evaluation.core.metrics.geval.LLMTestCase"
            ) as mock_test_case_class:
                mock_metric = MagicMock()
                mock_metric.score = 0.75
                mock_metric.reason = "Good match"
                mock_geval_class.return_value = mock_metric

                mock_test_case = MagicMock()
                mock_test_case_class.return_value = mock_test_case

                # Setup metric manager
                mock_metric_manager.get_metric_metadata.return_value = {
                    "criteria": "Compare against expected",
                    "evaluation_params": ["query", "response", "expected_response"],
                    "threshold": 0.7,
                }

                # Mock turn data with all optional fields
                turn_data = MagicMock()
                turn_data.query = "Test query"
                turn_data.response = "Test response"
                turn_data.expected_response = "Expected response"
                turn_data.contexts = ["Context 1", "Context 2"]

                conv_data = MagicMock()

                handler.evaluate(
                    metric_name="test_metric",
                    conv_data=conv_data,
                    _turn_idx=0,
                    turn_data=turn_data,
                    is_conversation=False,
                )

                # Verify test case was created with optional fields
                call_kwargs = mock_test_case_class.call_args[1]
                assert call_kwargs["input"] == "Test query"
                assert call_kwargs["actual_output"] == "Test response"
                assert call_kwargs["expected_output"] == "Expected response"
                assert call_kwargs["context"] == ["Context 1", "Context 2"]

    def test_evaluate_turn_none_score_returns_zero(self, handler, mock_metric_manager):
        """Test that None score from metric is converted to 0.0."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            mock_metric = MagicMock()
            mock_metric.score = None
            mock_metric.reason = "Could not evaluate"
            mock_geval_class.return_value = mock_metric

            mock_metric_manager.get_metric_metadata.return_value = {
                "criteria": "Test criteria",
                "threshold": 0.7,
            }

            turn_data = MagicMock()
            turn_data.query = "Test query"
            turn_data.response = "Test response"
            turn_data.expected_response = None
            turn_data.contexts = None

            conv_data = MagicMock()

            score, reason = handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                _turn_idx=0,
                turn_data=turn_data,
                is_conversation=False,
            )

            # Should return 0.0 when score is None
            assert score == 0.0
            assert reason == "Could not evaluate"

    def test_evaluate_turn_handles_exceptions(self, handler, mock_metric_manager):
        """Test that turn evaluation handles exceptions gracefully."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            mock_metric = MagicMock()
            mock_metric.measure.side_effect = ValueError("Test error")
            mock_geval_class.return_value = mock_metric

            mock_metric_manager.get_metric_metadata.return_value = {
                "criteria": "Test criteria",
                "threshold": 0.7,
            }

            turn_data = MagicMock()
            turn_data.query = "Test query"
            turn_data.response = "Test response"
            turn_data.expected_response = None
            turn_data.contexts = None

            conv_data = MagicMock()

            score, reason = handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                _turn_idx=0,
                turn_data=turn_data,
                is_conversation=False,
            )

            assert score is None
            assert "evaluation error" in reason.lower()
            assert "Test error" in reason

    def test_evaluate_turn_uses_default_params_when_none_provided(
        self, handler, mock_metric_manager
    ):
        """Test that default evaluation_params are used when none provided."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            mock_metric = MagicMock()
            mock_metric.score = 0.8
            mock_metric.reason = "Good"
            mock_geval_class.return_value = mock_metric

            # Config with no evaluation_params
            mock_metric_manager.get_metric_metadata.return_value = {
                "criteria": "Test criteria",
                "threshold": 0.7,
            }

            turn_data = MagicMock()
            turn_data.query = "Test query"
            turn_data.response = "Test response"
            turn_data.expected_response = None
            turn_data.contexts = None

            conv_data = MagicMock()

            handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                _turn_idx=0,
                turn_data=turn_data,
                is_conversation=False,
            )

            # Verify GEval was called with default params
            call_kwargs = mock_geval_class.call_args[1]
            assert call_kwargs["evaluation_params"] == [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ]

    def test_evaluate_conversation_success(self, handler, mock_metric_manager):
        """Test successful conversation-level evaluation."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            mock_metric = MagicMock()
            mock_metric.score = 0.90
            mock_metric.reason = "Conversation coherent"
            mock_geval_class.return_value = mock_metric

            mock_metric_manager.get_metric_metadata.return_value = {
                "criteria": "Conversation criteria",
                "evaluation_params": ["query", "response"],
                "threshold": 0.6,
            }

            # Mock conversation data with multiple turns
            turn1 = MagicMock()
            turn1.query = "Query 1"
            turn1.response = "Response 1"

            turn2 = MagicMock()
            turn2.query = "Query 2"
            turn2.response = "Response 2"

            conv_data = MagicMock()
            conv_data.turns = [turn1, turn2]

            score, reason = handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                _turn_idx=None,
                turn_data=None,
                is_conversation=True,
            )

            assert score == 0.90
            assert reason == "Conversation coherent"
            mock_metric.measure.assert_called_once()

    def test_evaluate_conversation_aggregates_turns(self, handler, mock_metric_manager):
        """Test that conversation evaluation properly aggregates turn data."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            with patch(
                "lightspeed_evaluation.core.metrics.geval.LLMTestCase"
            ) as mock_test_case_class:
                mock_metric = MagicMock()
                mock_metric.score = 0.85
                mock_metric.reason = "Good conversation"
                mock_geval_class.return_value = mock_metric

                mock_test_case = MagicMock()
                mock_test_case_class.return_value = mock_test_case

                mock_metric_manager.get_metric_metadata.return_value = {
                    "criteria": "Conversation flow",
                    "threshold": 0.7,
                }

                # Create multiple turns including one with None response
                turn1 = MagicMock()
                turn1.query = "First question"
                turn1.response = "First answer"

                turn2 = MagicMock()
                turn2.query = "Second question"
                turn2.response = "Second answer"

                turn3 = MagicMock()
                turn3.query = "Third question"
                turn3.response = None  # Test None response handling

                conv_data = MagicMock()
                conv_data.turns = [turn1, turn2, turn3]

                handler.evaluate(
                    metric_name="test_metric",
                    conv_data=conv_data,
                    _turn_idx=None,
                    turn_data=None,
                    is_conversation=True,
                )

                # Verify test case was created with aggregated input/output
                call_kwargs = mock_test_case_class.call_args[1]
                assert "Turn 1 - User: First question" in call_kwargs["input"]
                assert "Turn 2 - User: Second question" in call_kwargs["input"]
                assert "Turn 3 - User: Third question" in call_kwargs["input"]
                assert (
                    "Turn 1 - Assistant: First answer" in call_kwargs["actual_output"]
                )
                assert (
                    "Turn 2 - Assistant: Second answer" in call_kwargs["actual_output"]
                )
                assert "Turn 3 - Assistant:" in call_kwargs["actual_output"]

    def test_evaluate_conversation_with_evaluation_steps(
        self, handler, mock_metric_manager
    ):
        """Test that evaluation_steps are passed to GEval when provided."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            mock_metric = MagicMock()
            mock_metric.score = 0.88
            mock_metric.reason = "Follows steps"
            mock_geval_class.return_value = mock_metric

            mock_metric_manager.get_metric_metadata.return_value = {
                "criteria": "Multi-step evaluation",
                "evaluation_params": ["query", "response"],
                "evaluation_steps": [
                    "Check coherence",
                    "Verify context",
                    "Assess relevance",
                ],
                "threshold": 0.7,
            }

            turn1 = MagicMock()
            turn1.query = "Query 1"
            turn1.response = "Response 1"

            conv_data = MagicMock()
            conv_data.turns = [turn1]

            handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                _turn_idx=None,
                turn_data=None,
                is_conversation=True,
            )

            # Verify evaluation_steps were passed to GEval
            call_kwargs = mock_geval_class.call_args[1]
            assert call_kwargs["evaluation_steps"] == [
                "Check coherence",
                "Verify context",
                "Assess relevance",
            ]

    def test_evaluate_conversation_handles_exceptions(
        self, handler, mock_metric_manager
    ):
        """Test that conversation evaluation handles exceptions gracefully."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            mock_metric = MagicMock()
            mock_metric.measure.side_effect = RuntimeError("API error")
            mock_geval_class.return_value = mock_metric

            mock_metric_manager.get_metric_metadata.return_value = {
                "criteria": "Test criteria",
                "threshold": 0.7,
            }

            turn1 = MagicMock()
            turn1.query = "Query 1"
            turn1.response = "Response 1"

            conv_data = MagicMock()
            conv_data.turns = [turn1]

            score, reason = handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                _turn_idx=None,
                turn_data=None,
                is_conversation=True,
            )

            assert score is None
            assert "evaluation error" in reason.lower()
            assert "API error" in reason
