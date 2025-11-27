"""Unit tests for EvaluationPipeline."""

import pytest

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation.pipeline import EvaluationPipeline


@pytest.fixture
def mock_config_loader(mocker):
    """Create a mock config loader with system config."""
    loader = mocker.Mock(spec=ConfigLoader)

    config = SystemConfig()
    config.api.enabled = False
    config.output.output_dir = "/tmp/test_output"
    config.output.base_filename = "test"
    config.core.max_threads = 2

    loader.system_config = config
    return loader


@pytest.fixture
def sample_evaluation_data():
    """Create sample evaluation data."""
    turn1 = TurnData(
        turn_id="turn1",
        query="What is Python?",
        response="Python is a programming language.",
        contexts=["Python context"],
        turn_metrics=["ragas:faithfulness"],
    )
    conv_data = EvaluationData(
        conversation_group_id="conv1",
        turns=[turn1],
    )
    return [conv_data]


class TestEvaluationPipeline:
    """Unit tests for EvaluationPipeline."""

    def test_initialization_success(self, mock_config_loader, mocker):
        """Test successful pipeline initialization."""
        # Mock components
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.APIClient")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        pipeline = EvaluationPipeline(mock_config_loader)

        assert pipeline.config_loader == mock_config_loader
        assert pipeline.system_config is not None
        assert pipeline.output_dir == "/tmp/test_output"

    def test_initialization_without_config(self, mocker):
        """Test initialization fails without system config."""
        loader = mocker.Mock(spec=ConfigLoader)
        loader.system_config = None

        with pytest.raises(ValueError, match="SystemConfig must be loaded"):
            EvaluationPipeline(loader)

    def test_create_api_client_when_enabled(self, mock_config_loader, mocker):
        """Test API client creation when enabled."""
        mock_config_loader.system_config.api.enabled = True
        mock_config_loader.system_config.api.api_base = "http://test.com"
        mock_config_loader.system_config.api.endpoint_type = "test"

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mock_api_client = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIClient"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        pipeline = EvaluationPipeline(mock_config_loader)

        assert pipeline.api_client is not None
        mock_api_client.assert_called_once()

    def test_create_api_client_when_disabled(self, mock_config_loader, mocker):
        """Test no API client when disabled."""
        mock_config_loader.system_config.api.enabled = False

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        pipeline = EvaluationPipeline(mock_config_loader)

        assert pipeline.api_client is None

    def test_validate_data(self, mock_config_loader, sample_evaluation_data, mocker):
        """Test data validation."""
        mock_validator = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator"
        )
        mock_validator.return_value.validate_evaluation_data.return_value = True

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        pipeline = EvaluationPipeline(mock_config_loader)
        result = pipeline.validate_data(sample_evaluation_data)

        assert result is True
        mock_validator.return_value.validate_evaluation_data.assert_called_once_with(
            sample_evaluation_data
        )

    def test_run_evaluation_success(
        self, mock_config_loader, sample_evaluation_data, mocker
    ):
        """Test successful evaluation run."""
        # Mock all components
        mock_validator = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator"
        )
        mock_validator.return_value.validate_evaluation_data.return_value = True

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )

        # Mock conversation processor
        mock_processor = mocker.Mock()
        mock_result = EvaluationResult(
            conversation_group_id="conv1",
            turn_id="turn1",
            metric_identifier="ragas:faithfulness",
            score=0.85,
            result="PASS",
            threshold=0.7,
            reason="Good",
        )
        mock_processor.process_conversation.return_value = [mock_result]

        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor",
            return_value=mock_processor,
        )

        pipeline = EvaluationPipeline(mock_config_loader)
        results = pipeline.run_evaluation(sample_evaluation_data)

        assert len(results) == 1
        assert results[0].result == "PASS"

    def test_run_evaluation_validation_failure(
        self, mock_config_loader, sample_evaluation_data, mocker
    ):
        """Test evaluation fails on validation error."""
        mock_validator = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator"
        )
        mock_validator.return_value.validate_evaluation_data.return_value = False

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        pipeline = EvaluationPipeline(mock_config_loader)

        with pytest.raises(ValueError, match="Data validation failed"):
            pipeline.run_evaluation(sample_evaluation_data)

    def test_run_evaluation_saves_amended_data_when_api_enabled(
        self, mock_config_loader, sample_evaluation_data, mocker
    ):
        """Test amended data is saved when API is enabled."""
        mock_config_loader.system_config.api.enabled = True

        mock_validator = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator"
        )
        mock_validator.return_value.validate_evaluation_data.return_value = True

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.APIClient")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )

        mock_processor = mocker.Mock()
        mock_processor.process_conversation.return_value = []
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor",
            return_value=mock_processor,
        )

        mock_save = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.save_evaluation_data"
        )
        mock_save.return_value = "/tmp/amended.yaml"

        pipeline = EvaluationPipeline(mock_config_loader)
        pipeline.run_evaluation(sample_evaluation_data, "/tmp/original.yaml")

        mock_save.assert_called_once()

    def test_save_amended_data_handles_exception(
        self, mock_config_loader, sample_evaluation_data, mocker
    ):
        """Test save amended data handles exceptions gracefully."""
        mock_config_loader.system_config.api.enabled = True

        mock_validator = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator"
        )
        mock_validator.return_value.validate_evaluation_data.return_value = True

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.APIClient")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )

        mock_processor = mocker.Mock()
        mock_processor.process_conversation.return_value = []
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor",
            return_value=mock_processor,
        )

        # Mock save to raise exception
        mock_save = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.save_evaluation_data"
        )
        mock_save.side_effect = Exception("Save error")

        pipeline = EvaluationPipeline(mock_config_loader)
        # Should not raise, just log warning
        results = pipeline.run_evaluation(sample_evaluation_data, "/tmp/original.yaml")

        assert results is not None

    def test_close_with_api_client(self, mock_config_loader, mocker):
        """Test close method with API client."""
        mock_config_loader.system_config.api.enabled = True
        mock_config_loader.system_config.api.api_base = "http://test.com"
        mock_config_loader.system_config.api.endpoint_type = "test"

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mock_api_client_class = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIClient"
        )
        mock_api_client = mocker.Mock()
        mock_api_client_class.return_value = mock_api_client

        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        # Mock litellm.cache
        mock_litellm = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.litellm"
        )
        mock_cache = mocker.Mock()
        mock_litellm.cache = mock_cache

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.asyncio.run")

        pipeline = EvaluationPipeline(mock_config_loader)
        pipeline.close()

        mock_api_client.close.assert_called_once()

    def test_close_without_api_client(self, mock_config_loader, mocker):
        """Test close method without API client."""
        mock_config_loader.system_config.api.enabled = False

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        mock_litellm = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.litellm"
        )
        mock_litellm.cache = None

        pipeline = EvaluationPipeline(mock_config_loader)
        # Should not raise any errors
        pipeline.close()

    def test_output_dir_override(self, mock_config_loader, mocker):
        """Test output directory can be overridden."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.DataValidator")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.APIDataAmender"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.EvaluationErrorHandler"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ScriptExecutionManager"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.MetricsEvaluator"
        )
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.ConversationProcessor"
        )

        pipeline = EvaluationPipeline(mock_config_loader, output_dir="/custom/output")

        assert pipeline.output_dir == "/custom/output"
