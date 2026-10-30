# pylint: disable=protected-access

"""Unit tests for EvaluationPipeline."""

from typing import Any

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    TurnData,
)
from lightspeed_evaluation.core.models.agents import AgentsConfig
from lightspeed_evaluation.core.models.system import SystemConfig
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
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
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
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

    def test_create_default_driver(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test default driver is created via registry."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mock_registry_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
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

        mock_registry_cls.return_value.create_driver.assert_called_once()
        assert pipeline._default_driver is not None

    def test_create_default_driver_fallback_config(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test default driver uses disabled http_api fallback when no agents config."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mock_registry_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
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

        EvaluationPipeline(mock_config_loader)

        mock_registry_cls.return_value.create_driver.assert_called_once_with(
            {"type": "http_api"}, enabled=False
        )

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
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
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

    def test_run_evaluation_saves_amended_data_when_agents_enabled(
        self,
        mock_config_loader: ConfigLoader,
        sample_evaluation_data: list[EvaluationData],
        mocker: MockerFixture,
    ) -> None:
        """Test amended data is saved when agents config is enabled."""
        assert mock_config_loader.system_config is not None
        mock_config_loader.system_config.agents = AgentsConfig(enabled=True)

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mock_registry_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
        )
        mock_driver = mocker.Mock()
        mock_driver.enabled = True
        mock_registry_cls.return_value.create_driver.return_value = mock_driver

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
        mock_config_loader.system_config.agents = AgentsConfig(enabled=True)

        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mock_registry_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
        )
        mock_driver = mocker.Mock()
        mock_driver.enabled = True
        mock_registry_cls.return_value.create_driver.return_value = mock_driver

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

    def test_close_calls_driver_close(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test close method calls driver close."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mock_registry_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
        )
        mock_driver = mocker.Mock()
        mock_registry_cls.return_value.create_driver.return_value = mock_driver

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

        mock_driver.close.assert_called_once()

    def test_close_without_cache(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test close method when no litellm cache exists."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
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

    def test_close_handles_already_disconnected_cache(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test close handles already-disconnected cache gracefully."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
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

        # Mock litellm.cache with a disconnect that raises
        mock_litellm = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.litellm"
        )
        mock_cache = mocker.Mock()
        mock_litellm.cache = mock_cache

        mock_asyncio_run = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.asyncio.run"
        )
        mock_asyncio_run.side_effect = RuntimeError("already disconnected")

        pipeline = EvaluationPipeline(mock_config_loader)
        # Should not raise even though disconnect() fails
        pipeline.close()

        # Cache should be set to None after close
        assert mock_litellm.cache is None

    def test_output_dir_override(
        self, mock_config_loader: ConfigLoader, mocker: MockerFixture
    ) -> None:
        """Test output directory can be overridden."""
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.pipeline.MetricManager")
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.AgentDriverRegistry"
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


class TestResolveDriverTraceSource:
    """Tests for trace_source validation at the eval-data override boundary."""

    @staticmethod
    def _make_pipeline(
        mocker: MockerFixture,
        trace_sources: dict[str, Any],
    ) -> Any:
        """Build a bare pipeline with only the fields the resolver touches."""
        pipeline = EvaluationPipeline.__new__(EvaluationPipeline)
        pipeline._default_driver = mocker.Mock()
        pipeline._registry = mocker.Mock()
        pipeline.system_config = SystemConfig.model_validate(
            {
                "agents": {
                    "default": {"agent": ["a1"]},
                    "a1": {"type": "http_api"},
                },
                "trace_sources": trace_sources,
            }
        )
        return pipeline

    @staticmethod
    def _conv(agent_config: dict[str, Any]) -> EvaluationData:
        """Evaluation data with a single turn and an agent config override."""
        return EvaluationData(
            conversation_group_id="g1",
            turns=[TurnData(turn_id="t1", query="q")],
            agent_config=agent_config,
        )

    def test_unknown_trace_source_override_rejected(
        self, mocker: MockerFixture
    ) -> None:
        """Eval-data override with unknown trace_source fails at driver creation."""
        pipeline = self._make_pipeline(mocker, {"lf": {"type": "langfuse"}})
        conv_data = self._conv({"trace_source": "unknown"})
        with pytest.raises(ConfigurationError, match="unknown trace_source"):
            pipeline._resolve_driver_for_conversation(conv_data)

    def test_known_trace_source_override_accepted(self, mocker: MockerFixture) -> None:
        """Eval-data override referencing a registered source creates a driver."""
        pipeline = self._make_pipeline(mocker, {"lf": {"type": "langfuse"}})
        conv_data = self._conv({"trace_source": "lf"})
        _driver, is_per_conversation = pipeline._resolve_driver_for_conversation(
            conv_data
        )
        assert is_per_conversation
        pipeline._registry.create_driver.assert_called_once()

    def test_none_trace_source_override_skips_validation(
        self, mocker: MockerFixture
    ) -> None:
        """The 'none' sentinel disables capture and is accepted."""
        pipeline = self._make_pipeline(mocker, {"lf": {"type": "langfuse"}})
        conv_data = self._conv({"trace_source": "none"})
        pipeline._resolve_driver_for_conversation(conv_data)
        pipeline._registry.create_driver.assert_called_once()

    def test_non_string_trace_source_override_rejected(
        self, mocker: MockerFixture
    ) -> None:
        """A non-string trace_source override fails with a clear error."""
        pipeline = self._make_pipeline(mocker, {"lf": {"type": "langfuse"}})
        conv_data = self._conv({"trace_source": ["lf"]})
        with pytest.raises(ConfigurationError, match="must be a string"):
            pipeline._validate_trace_source_reference(conv_data)

    def test_validation_runs_before_any_work_submitted(
        self, mocker: MockerFixture
    ) -> None:
        """A bad reference fails validation before any conversation starts."""
        pipeline = self._make_pipeline(mocker, {"lf": {"type": "langfuse"}})
        good = self._conv({"trace_source": "lf"})
        good.conversation_group_id = "good"
        bad = self._conv({"trace_source": "unknown"})
        bad.conversation_group_id = "bad"

        executor_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.pipeline.concurrent"
            ".futures.ThreadPoolExecutor"
        )

        with pytest.raises(ConfigurationError, match="unknown trace_source"):
            pipeline._process_eval_data([good, bad])

        executor_cls.assert_not_called()

    @pytest.mark.parametrize("sentinel", [None, "", "none"])
    def test_disable_sentinels_accepted(
        self, sentinel: Any, mocker: MockerFixture
    ) -> None:
        """None, empty string, and 'none' all disable capture without error."""
        pipeline = self._make_pipeline(mocker, {"lf": {"type": "langfuse"}})
        conv_data = self._conv({"trace_source": sentinel})
        pipeline._validate_trace_source_reference(conv_data)

    @pytest.mark.parametrize("invalid", [[], False, 0])
    def test_falsey_non_sentinels_rejected(
        self, invalid: Any, mocker: MockerFixture
    ) -> None:
        """Falsey values that are not sentinels fail with a clear error."""
        pipeline = self._make_pipeline(mocker, {"lf": {"type": "langfuse"}})
        conv_data = self._conv({"trace_source": invalid})
        with pytest.raises(ConfigurationError, match="must be a string"):
            pipeline._validate_trace_source_reference(conv_data)
