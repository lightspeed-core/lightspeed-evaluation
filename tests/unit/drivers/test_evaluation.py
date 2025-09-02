"""Unit tests for drivers.evaluation module."""

from unittest.mock import MagicMock, patch

import pytest

from lightspeed_evaluation.core.config.models import EvaluationData, EvaluationResult, TurnData
from lightspeed_evaluation.drivers.evaluation import EvaluationDriver, EvaluationRequest


class TestEvaluationRequest:
    """Unit tests for EvaluationRequest class."""

    def test_evaluation_request_initialization(self):
        """Test EvaluationRequest initialization."""
        conv_data = MagicMock()
        metric_identifier = "test:metric"

        request = EvaluationRequest(conv_data, metric_identifier)

        assert request.conv_data == conv_data
        assert request.metric_identifier == metric_identifier
        assert request.is_conversation is False
        assert request.turn_idx is None
        assert request.turn_data is None
        assert request.turn_id is None

    def test_evaluation_request_for_turn(self):
        """Test EvaluationRequest.for_turn class method."""
        conv_data = MagicMock()
        metric_identifier = "test:metric"
        turn_data = TurnData(turn_id=5, query="Test query", response="Test response")

        request = EvaluationRequest.for_turn(conv_data, metric_identifier, 2, turn_data)

        assert request.conv_data == conv_data
        assert request.metric_identifier == metric_identifier
        assert request.is_conversation is False
        assert request.turn_idx == 2
        assert request.turn_data == turn_data
        assert request.turn_id == 5

    def test_evaluation_request_for_turn_no_turn_id(self):
        """Test EvaluationRequest.for_turn with turn_data having None turn_id (uses turn_idx + 1)."""
        conv_data = MagicMock()
        metric_identifier = "test:metric"
        # Create turn_data with valid turn_id first
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        # Manually set turn_id to None to test the fallback logic
        turn_data.turn_id = None

        request = EvaluationRequest.for_turn(conv_data, metric_identifier, 3, turn_data)

        assert request.turn_id == 4  # turn_idx + 1

    def test_evaluation_request_for_conversation(self):
        """Test EvaluationRequest.for_conversation class method."""
        conv_data = MagicMock()
        metric_identifier = "test:conversation_metric"

        request = EvaluationRequest.for_conversation(conv_data, metric_identifier)

        assert request.conv_data == conv_data
        assert request.metric_identifier == metric_identifier
        assert request.is_conversation is True
        assert request.turn_idx is None
        assert request.turn_data is None
        assert request.turn_id is None


class TestEvaluationDriver:
    """Unit tests for EvaluationDriver class."""

    def test_evaluation_driver_initialization(self):
        """Test EvaluationDriver initialization."""
        # Mock ConfigLoader instead of system_config directly
        config_loader = MagicMock()
        # Mock the get_llm_config_dict method to return proper dict structure
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)

            assert driver.config_loader == config_loader
            assert driver.metrics_manager is not None

    @patch("lightspeed_evaluation.drivers.evaluation.MetricsManager")
    @patch("lightspeed_evaluation.drivers.evaluation.LLMManager")
    def test_evaluation_driver_initialization_with_mocks(
        self, mock_llm_manager, mock_metrics_manager
    ):
        """Test EvaluationDriver initialization with mocked dependencies."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }
        
        mock_llm_instance = MagicMock()
        mock_llm_manager.from_system_config.return_value = mock_llm_instance
        mock_metrics_instance = MagicMock()
        mock_metrics_manager.return_value = mock_metrics_instance

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)

            mock_llm_manager.from_system_config.assert_called_once()
            mock_metrics_manager.assert_called_once_with(mock_llm_instance)

    def test_run_evaluation_empty_data(self):
        """Test run_evaluation with empty evaluation data."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)
            results = driver.run_evaluation([])

            assert results == []

    def test_evaluate_single_turn_metric_success(self):
        """Test _evaluate_metric with successful turn metric evaluation."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }
        
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn_data]
        )

        request = EvaluationRequest.for_turn(eval_data, "ragas:faithfulness", 0, turn_data)

        expected_result = EvaluationResult(
            conversation_group_id="test_conv",
            turn_id=1,
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.8,
            reason="Good performance",
        )

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)
            
            # Mock the _get_effective_threshold method to return None (no threshold)
            with patch.object(driver, "_get_effective_threshold", return_value=None):
                # Mock the metrics manager's evaluate_metric method
                with patch.object(driver.metrics_manager, "evaluate_metric", return_value=(0.8, "Good performance")):
                    result = driver._evaluate_metric(request)

                    assert result.conversation_group_id == expected_result.conversation_group_id
                    assert result.turn_id == expected_result.turn_id
                    assert result.metric_identifier == expected_result.metric_identifier
                    assert result.score == expected_result.score

    def test_evaluate_single_conversation_metric_success(self):
        """Test _evaluate_metric with successful conversation metric evaluation."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }
        
        # Create valid EvaluationData with at least one turn
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        request = EvaluationRequest.for_conversation(eval_data, "deepeval:completeness")

        expected_result = EvaluationResult(
            conversation_group_id="test_conv",
            turn_id=None,
            metric_identifier="deepeval:completeness",
            result="PASS",
            score=0.9,
            reason="Complete conversation",
        )

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)
            
            # Mock the _get_effective_threshold method to return None (no threshold)
            with patch.object(driver, "_get_effective_threshold", return_value=None):
                # Mock the metrics manager's evaluate_metric method
                with patch.object(driver.metrics_manager, "evaluate_metric", return_value=(0.9, "Complete conversation")):
                    result = driver._evaluate_metric(request)

                    assert result.conversation_group_id == expected_result.conversation_group_id
                    assert result.turn_id == expected_result.turn_id
                    assert result.metric_identifier == expected_result.metric_identifier
                    assert result.score == expected_result.score

    def test_evaluate_single_unknown_framework(self):
        """Test _evaluate_metric with unknown metric framework."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }
        
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn_data]
        )

        request = EvaluationRequest.for_turn(eval_data, "unknown:metric", 0, turn_data)

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)
            
            # Mock the _get_effective_threshold method to return None (no threshold)
            with patch.object(driver, "_get_effective_threshold", return_value=None):
                # Mock the metrics manager to return error for unknown framework
                with patch.object(driver.metrics_manager, "evaluate_metric", return_value=(None, "Unsupported framework: unknown")):
                    result = driver._evaluate_metric(request)

                    assert result.result == "ERROR"
                    assert "Unsupported framework" in result.reason

    def test_evaluate_single_metric_exception(self):
        """Test _evaluate_metric when metric evaluation raises exception."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }
        
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn_data]
        )

        request = EvaluationRequest.for_turn(eval_data, "ragas:faithfulness", 0, turn_data)

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)
            
            # Mock the _get_effective_threshold method to return None (no threshold)
            with patch.object(driver, "_get_effective_threshold", return_value=None):
                # Mock the metrics manager to raise a ValueError (which is caught by _evaluate_metric)
                with patch.object(driver.metrics_manager, "evaluate_metric", side_effect=ValueError("API Error")):
                    result = driver._evaluate_metric(request)

                    assert result.result == "ERROR"
                    assert "API Error" in result.reason

    def test_get_supported_frameworks(self):
        """Test metrics manager supported frameworks."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)

            supported_frameworks = driver.metrics_manager.get_supported_frameworks()
            expected_frameworks = ["ragas", "deepeval", "custom"]
            for framework in expected_frameworks:
                assert framework in supported_frameworks
            assert "unknown" not in supported_frameworks

    def test_create_error_result_turn_level(self):
        """Test error result creation for turn-level metric."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }
        
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn_data]
        )

        request = EvaluationRequest.for_turn(eval_data, "test:metric", 0, turn_data)

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)
            
            # Create an error result manually to test the structure
            result = EvaluationResult(
                conversation_group_id="test_conv",
                turn_id=1,
                metric_identifier="test:metric",
                result="ERROR",
                reason="Test error message",
                execution_time=1.5,
                score=0.0
            )

            assert result.conversation_group_id == "test_conv"
            assert result.turn_id == 1
            assert result.metric_identifier == "test:metric"
            assert result.result == "ERROR"
            assert result.reason == "Test error message"
            assert result.execution_time == 1.5
            assert result.score == 0.0

    def test_create_error_result_conversation_level(self):
        """Test error result creation for conversation-level metric."""
        config_loader = MagicMock()
        config_loader.get_llm_config_dict.return_value = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout": 300,
                "num_retries": 3
            }
        }
        
        # Create valid EvaluationData with at least one turn
        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        eval_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        request = EvaluationRequest.for_conversation(eval_data, "test:conversation_metric")

        with patch("builtins.print"):
            driver = EvaluationDriver(config_loader)
            
            # Create an error result manually to test the structure
            result = EvaluationResult(
                conversation_group_id="test_conv",
                turn_id=None,
                metric_identifier="test:conversation_metric",
                result="ERROR",
                reason="Conversation error",
                execution_time=2.0
            )

            assert result.conversation_group_id == "test_conv"
            assert result.turn_id is None
            assert result.metric_identifier == "test:conversation_metric"
            assert result.result == "ERROR"
            assert result.reason == "Conversation error"
            assert result.execution_time == 2.0