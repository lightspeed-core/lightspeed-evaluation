"""Tests for GEval metrics handler."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

import pytest
from deepeval.test_case import LLMTestCaseParams

from lightspeed_evaluation.core.metrics.geval import GEvalHandler


class TestGEvalHandler:
    """Test cases for GEvalHandler class."""

    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset class-level registry before each test."""
        GEvalHandler._registry = None
        GEvalHandler._registry_path = None
        yield
        GEvalHandler._registry = None
        GEvalHandler._registry_path = None

    def test_initialization_with_mock_manager(self):
        """Test GEvalHandler initialization with mock LLM manager."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        assert handler.deepeval_llm_manager == mock_manager
        assert (
            GEvalHandler._registry is not None
        )  # Should be initialized (empty or loaded)

    def test_registry_loading_from_file(self):
        """Test loading registry from a YAML file."""
        # Create temporary YAML file with test metrics
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """\
test_metric:
  criteria: "Test criteria"
  evaluation_params:
    - input
    - actual_output
  threshold: 0.7
  """
            )
            temp_path = f.name

        try:
            mock_manager = MagicMock()
            GEvalHandler(deepeval_llm_manager=mock_manager, registry_path=temp_path)

            assert GEvalHandler._registry is not None
            assert "test_metric" in GEvalHandler._registry
            assert GEvalHandler._registry["test_metric"]["criteria"] == "Test criteria"
            assert GEvalHandler._registry["test_metric"]["threshold"] == 0.7
        finally:
            Path(temp_path).unlink()

    def test_convert_evaluation_params_valid(self):
        """Test conversion of valid evaluation params to enum."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        params = ["input", "actual_output", "expected_output"]
        result = handler._convert_evaluation_params(params)

        assert result is not None
        assert len(result) == 3
        assert LLMTestCaseParams.INPUT in result
        assert LLMTestCaseParams.ACTUAL_OUTPUT in result
        assert LLMTestCaseParams.EXPECTED_OUTPUT in result

    def test_convert_evaluation_params_invalid(self):
        """Test conversion returns None for invalid params."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        params = ["invalid_param", "another_invalid"]
        result = handler._convert_evaluation_params(params)

        assert result is None

    def test_convert_evaluation_params_empty(self):
        """Test conversion with empty params list."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        result = handler._convert_evaluation_params([])

        assert result is None

    def test_get_geval_config_from_registry(self):
        """Test retrieving config from registry."""
        # Create temporary registry
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(
                """
test_metric:
  criteria: "Registry criteria"
  threshold: 0.8
"""
            )
            temp_path = f.name

        try:
            mock_manager = MagicMock()
            handler = GEvalHandler(
                deepeval_llm_manager=mock_manager, registry_path=temp_path
            )

            # Mock conversation data without metadata
            conv_data = MagicMock()
            conv_data.conversation_metrics_metadata = None

            config = handler._get_geval_config(
                metric_name="test_metric",
                conv_data=conv_data,
                turn_data=None,
                is_conversation=True,
            )

            assert config is not None
            assert config["criteria"] == "Registry criteria"
            assert config["threshold"] == 0.8
        finally:
            Path(temp_path).unlink()

    def test_get_geval_config_from_conversation_metadata(self):
        """Test retrieving config from conversation metadata (overrides registry)."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        # Mock conversation data with metadata
        conv_data = MagicMock()
        conv_data.conversation_metrics_metadata = {
            "geval:test_metric": {"criteria": "Runtime criteria", "threshold": 0.9}
        }

        config = handler._get_geval_config(
            metric_name="test_metric",
            conv_data=conv_data,
            turn_data=None,
            is_conversation=True,
        )

        assert config is not None
        assert config["criteria"] == "Runtime criteria"
        assert config["threshold"] == 0.9

    def test_get_geval_config_from_turn_metadata(self):
        """Test retrieving config from turn metadata (highest priority)."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        # Mock conversation and turn data
        conv_data = MagicMock()
        conv_data.conversation_metrics_metadata = {
            "geval:test_metric": {"criteria": "Conv criteria", "threshold": 0.7}
        }

        turn_data = MagicMock()
        turn_data.turn_metrics_metadata = {
            "geval:test_metric": {"criteria": "Turn criteria", "threshold": 0.95}
        }

        config = handler._get_geval_config(
            metric_name="test_metric",
            conv_data=conv_data,
            turn_data=turn_data,
            is_conversation=False,  # Turn-level
        )

        assert config is not None
        assert config["criteria"] == "Turn criteria"
        assert config["threshold"] == 0.95

    def test_get_geval_config_not_found(self):
        """Test handling when config is not found anywhere."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        conv_data = MagicMock()
        conv_data.conversation_metrics_metadata = None

        config = handler._get_geval_config(
            metric_name="nonexistent_metric",
            conv_data=conv_data,
            turn_data=None,
            is_conversation=True,
        )

        assert config is None

    def test_evaluate_missing_config(self):
        """Test evaluation with missing configuration."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        conv_data = MagicMock()
        conv_data.conversation_metrics_metadata = None

        score, reason = handler.evaluate(
            metric_name="nonexistent",
            conv_data=conv_data,
            turn_idx=0,
            turn_data=None,
            is_conversation=True,
        )

        assert score is None
        assert "configuration not found" in reason.lower()

    def test_evaluate_missing_criteria(self):
        """Test evaluation with config missing criteria."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        conv_data = MagicMock()
        conv_data.conversation_metrics_metadata = {
            "geval:test_metric": {
                "threshold": 0.8
                # Missing criteria
            }
        }

        score, reason = handler.evaluate(
            metric_name="test_metric",
            conv_data=conv_data,
            turn_idx=0,
            turn_data=None,
            is_conversation=True,
        )

        assert score is None
        assert "criteria" in reason.lower()

    def test_evaluate_turn_missing_turn_data(self):
        """Test turn-level evaluation with missing turn data."""
        mock_manager = MagicMock()
        handler = GEvalHandler(deepeval_llm_manager=mock_manager)

        conv_data = MagicMock()
        conv_data.conversation_metrics_metadata = {
            "geval:test_metric": {"criteria": "Test criteria"}
        }

        score, reason = handler.evaluate(
            metric_name="test_metric",
            conv_data=conv_data,
            turn_idx=0,
            turn_data=None,  # Missing turn data
            is_conversation=False,  # Turn-level
        )

        assert score is None
        assert "turn data required" in reason.lower()

    def test_evaluate_turn_success(self):
        """Test successful turn-level evaluation."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            # Mock GEval metric
            mock_metric = MagicMock()
            mock_metric.score = 0.85
            mock_metric.reason = "Test passed"
            mock_geval_class.return_value = mock_metric

            # Mock LLM manager
            mock_llm_manager = MagicMock()
            mock_llm = MagicMock()
            mock_llm_manager.get_llm.return_value = mock_llm

            handler = GEvalHandler(deepeval_llm_manager=mock_llm_manager)

            # Mock turn data
            turn_data = MagicMock()
            turn_data.query = "Test query"
            turn_data.response = "Test response"
            turn_data.expected_response = None
            turn_data.contexts = None

            # Mock conv data with config
            conv_data = MagicMock()
            conv_data.conversation_metrics_metadata = {
                "geval:test_metric": {
                    "criteria": "Test criteria",
                    "evaluation_steps": ["Step 1", "Step 2"],
                    "threshold": 0.7,
                }
            }

            score, reason = handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                turn_idx=0,
                turn_data=turn_data,
                is_conversation=False,
            )

            assert score == 0.85
            assert reason == "Test passed"
            mock_metric.measure.assert_called_once()

    def test_evaluate_conversation_success(self):
        """Test successful conversation-level evaluation."""
        with patch(
            "lightspeed_evaluation.core.metrics.geval.GEval"
        ) as mock_geval_class:
            # Mock GEval metric
            mock_metric = MagicMock()
            mock_metric.score = 0.90
            mock_metric.reason = "Conversation coherent"
            mock_geval_class.return_value = mock_metric

            # Mock LLM manager
            mock_llm_manager = MagicMock()
            mock_llm = MagicMock()
            mock_llm_manager.get_llm.return_value = mock_llm

            handler = GEvalHandler(deepeval_llm_manager=mock_llm_manager)

            # Mock conversation data
            turn1 = MagicMock()
            turn1.query = "Query 1"
            turn1.response = "Response 1"

            turn2 = MagicMock()
            turn2.query = "Query 2"
            turn2.response = "Response 2"

            conv_data = MagicMock()
            conv_data.turns = [turn1, turn2]
            conv_data.conversation_metrics_metadata = {
                "geval:test_metric": {
                    "criteria": "Conversation criteria",
                    "threshold": 0.6,
                }
            }

            score, reason = handler.evaluate(
                metric_name="test_metric",
                conv_data=conv_data,
                turn_idx=None,
                turn_data=None,
                is_conversation=True,
            )

            assert score == 0.90
            assert reason == "Conversation coherent"
            mock_metric.measure.assert_called_once()
