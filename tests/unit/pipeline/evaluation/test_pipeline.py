"""Unit tests for EvaluationPipeline."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
)
from lightspeed_evaluation.core.system.loader import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation.pipeline import EvaluationPipeline


class TestEvaluationPipeline:
    """Unit tests for EvaluationPipeline."""

    def test_initialization_success(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test successful pipeline initialization."""
        # Mock components
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

    def test_initialization_without_config(self, mocker: MockerFixture) -> None:
        """Test initialization fails without system config."""
        loader = mocker.Mock(spec=ConfigLoader)
        loader.system_config = None

        with pytest.raises(ValueError, match="SystemConfig must be loaded"):
            EvaluationPipeline(loader)

    def test_create_api_client_when_enabled(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test API client creation when enabled."""
        assert mock_config_loader.system_config is not None
        mock_config_loader.system_config.api.enabled = True
        mock_config_loader.system_config.api.api_base = "http://test.com"
        mock_config_loader.system_config.api.endpoint_type = "test"

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

    def test_create_api_client_when_disabled(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test no API client when disabled."""
        assert mock_config_loader.system_config is not None
        mock_config_loader.system_config.api.enabled = False

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

    def test_run_evaluation_success(
        self,
        mock_config_loader: ConfigLoader,
        sample_evaluation_data: list[EvaluationData],
        mocker: MockerFixture,
    ) -> None:
        """Test successful evaluation run."""
        # Mock all components
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

    def test_run_evaluation_saves_amended_data_when_api_enabled(
        self,
        mock_config_loader: ConfigLoader,
        sample_evaluation_data: list[EvaluationData],
        mocker: MockerFixture,
    ) -> None:
        """Test amended data is saved when API is enabled."""
        assert mock_config_loader.system_config is not None
        mock_config_loader.system_config.api.enabled = True

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
        self,
        mock_config_loader: ConfigLoader,
        sample_evaluation_data: list[EvaluationData],
        mocker: MockerFixture,
    ) -> None:
        """Test save amended data handles exceptions gracefully."""
        assert mock_config_loader.system_config is not None
        mock_config_loader.system_config.api.enabled = True

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

    def test_close_with_api_client(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test close method with API client."""
        assert mock_config_loader.system_config is not None
        mock_config_loader.system_config.api.enabled = True
        mock_config_loader.system_config.api.api_base = "http://test.com"
        mock_config_loader.system_config.api.endpoint_type = "test"

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

    def test_close_without_api_client(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test close method without API client."""
        assert mock_config_loader.system_config is not None
        mock_config_loader.system_config.api.enabled = False

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

    def test_output_dir_override(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test output directory can be overridden."""
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
