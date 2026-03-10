"""Unit tests for LLM Manager."""

import logging

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import (
    LLMConfig,
    SystemConfig,
    LLMPoolConfig,
    JudgePanelConfig,
)
from lightspeed_evaluation.core.models.system import (
    LLMDefaultsConfig,
    LLMParametersConfig,
    LLMProviderConfig,
)
from lightspeed_evaluation.core.llm.manager import LLMManager


class TestLLMManager:
    """Tests for LLMManager."""

    def test_initialization_openai(
        self, basic_llm_config: LLMConfig, mocker: MockerFixture
    ) -> None:
        """Test initialization with OpenAI provider."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(basic_llm_config)

        assert manager.model_name == "gpt-4"
        assert manager.config.provider == "openai"

    def test_initialization_azure(self, mocker: MockerFixture) -> None:
        """Test initialization with Azure provider."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")
        mocker.patch.dict("os.environ", {})

        manager = LLMManager(config)

        assert "azure" in manager.model_name

    def test_initialization_azure_with_deployment(self, mocker: MockerFixture) -> None:
        """Test initialization with Azure deployment name."""
        config = LLMConfig(
            provider="azure",
            model="gpt-4",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")
        mocker.patch.dict("os.environ", {"AZURE_DEPLOYMENT_NAME": "my-deployment"})

        manager = LLMManager(config)

        assert manager.model_name == "azure/my-deployment"

    def test_initialization_watsonx(self, mocker: MockerFixture) -> None:
        """Test initialization with WatsonX provider."""
        config = LLMConfig(
            provider="watsonx",
            model="ibm/granite-13b",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)

        assert manager.model_name == "watsonx/ibm/granite-13b"

    def test_initialization_anthropic(self, mocker: MockerFixture) -> None:
        """Test initialization with Anthropic provider."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-opus",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)

        assert manager.model_name == "anthropic/claude-3-opus"

    def test_initialization_gemini(self, mocker: MockerFixture) -> None:
        """Test initialization with Gemini provider."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-pro",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)

        assert manager.model_name == "gemini/gemini-pro"

    def test_initialization_vertex(self, mocker: MockerFixture) -> None:
        """Test initialization with Vertex AI provider."""
        config = LLMConfig(
            provider="vertex",
            model="gemini-pro",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)

        assert manager.model_name == "gemini-pro"

    def test_initialization_ollama(self, mocker: MockerFixture) -> None:
        """Test initialization with Ollama provider."""
        config = LLMConfig(
            provider="ollama",
            model="llama2",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)

        assert manager.model_name == "ollama/llama2"

    def test_initialization_hosted_vllm(self, mocker: MockerFixture) -> None:
        """Test initialization with hosted vLLM provider."""
        config = LLMConfig(
            provider="hosted_vllm",
            model="mistral-7b",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)

        assert manager.model_name == "hosted_vllm/mistral-7b"

    def test_initialization_generic_provider(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test initialization with unknown/generic provider."""
        config = LLMConfig(
            provider="custom_provider",
            model="custom-model",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        with caplog.at_level(logging.WARNING):
            manager = LLMManager(config)

        # Should construct generic model name
        assert manager.model_name == "custom_provider/custom-model"

        # Should log warning about generic provider
        assert any("generic" in record.message.lower() for record in caplog.records)

    def test_get_model_name(
        self, basic_llm_config: LLMConfig, mocker: MockerFixture
    ) -> None:
        """Test get_model_name method."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(basic_llm_config)

        assert manager.get_model_name() == "gpt-4"

    def test_get_llm_params(
        self, basic_llm_config: LLMConfig, mocker: MockerFixture
    ) -> None:
        """Test get_llm_params method."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(basic_llm_config)
        params = manager.get_llm_params()

        assert params["model"] == "gpt-4"
        assert params["temperature"] == 0.0
        assert params["max_completion_tokens"] == 512
        assert params["timeout"] == 60
        assert params["num_retries"] == 3

    def test_get_config(
        self, basic_llm_config: LLMConfig, mocker: MockerFixture
    ) -> None:
        """Test get_config method."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(basic_llm_config)
        config = manager.get_config()

        assert config == basic_llm_config
        assert config.provider == "openai"
        assert config.model == "gpt-4"

    def test_from_system_config(self, mocker: MockerFixture) -> None:
        """Test creating manager from SystemConfig."""
        system_config = SystemConfig()
        system_config.llm = LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.5,
        )

        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager.from_system_config(system_config)

        assert manager.config.model == "gpt-3.5-turbo"
        assert manager.config.temperature == 0.5

    def test_from_llm_config(
        self, basic_llm_config: LLMConfig, mocker: MockerFixture
    ) -> None:
        """Test creating manager from LLMConfig."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager.from_llm_config(basic_llm_config)

        assert manager.config == basic_llm_config

    def test_llm_params_with_custom_values(self, mocker: MockerFixture) -> None:
        """Test LLM params with custom configuration values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1024,
            timeout=120,
            num_retries=5,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)
        params = manager.get_llm_params()

        assert params["temperature"] == 0.7
        assert params["max_completion_tokens"] == 1024
        assert params["timeout"] == 120
        assert params["num_retries"] == 5

    def test_initialization_logs_message(
        self,
        basic_llm_config: LLMConfig,
        mocker: MockerFixture,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that initialization logs configuration message."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        with caplog.at_level(logging.INFO):
            LLMManager(basic_llm_config)

        # Should log LLM manager info
        assert any("LLM Manager" in record.message for record in caplog.records)
        assert any("openai" in record.message for record in caplog.records)
        assert any("gpt-4" in record.message for record in caplog.records)


def _create_llm_pool_with_judges(
    judges: list[tuple[str, str]],
    enabled_metrics: list[str] | None = None,
) -> tuple[LLMPoolConfig, JudgePanelConfig]:
    """Helper to create LLMPoolConfig and JudgePanelConfig from judge list.

    Args:
        judges: List of (provider, model) tuples.
        enabled_metrics: Optional list of metrics to enable for panel.
    """
    models: dict[str, LLMProviderConfig] = {}
    for provider, model in judges:
        models[model] = LLMProviderConfig(provider=provider)

    pool = LLMPoolConfig(
        defaults=LLMDefaultsConfig(
            parameters=LLMParametersConfig(temperature=0.0, max_completion_tokens=512)
        ),
        models=models,
    )
    judge_ids = [model for _, model in judges]
    panel = JudgePanelConfig(judges=judge_ids, enabled_metrics=enabled_metrics)
    return pool, panel


class TestLLMManagerJudgePanel:
    """Tests for LLMManager judge panel functionality."""

    def test_without_judge_panel(self, mocker: MockerFixture) -> None:
        """Test LLMManager without judge panel configured."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")
        manager = LLMManager(LLMConfig(provider="openai", model="gpt-4o-mini"))

        assert not manager.has_judge_panel()
        assert len(manager.judge_managers) == 0
        assert len(manager.get_judge_managers()) == 1
        assert manager.get_primary_judge() is manager
        assert not manager.should_use_panel_for_metric("ragas:faithfulness")

    def test_with_judge_panel(self, mocker: MockerFixture) -> None:
        """Test LLMManager with judge panel configured."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        pool, panel = _create_llm_pool_with_judges(
            [
                ("openai", "gpt-4o-mini"),
                ("openai", "gpt-4o"),
                ("gemini", "gemini-2.0-flash-exp"),
            ]
        )
        system_config = SystemConfig(llm_pool=pool, judge_panel=panel)
        manager = LLMManager.from_system_config(system_config)

        # Panel detected
        assert manager.has_judge_panel()
        assert len(manager.judge_managers) == 3

        # Judge managers
        judges = manager.get_judge_managers()
        assert len(judges) == 3
        assert judges[0].config.model == "gpt-4o-mini"
        assert judges[1].config.model == "gpt-4o"
        assert judges[2].config.model == "gemini-2.0-flash-exp"

        # Primary judge is first
        assert manager.get_primary_judge().config.model == "gpt-4o-mini"

    def test_should_use_panel_with_enabled_metrics(self, mocker: MockerFixture) -> None:
        """Test should_use_panel with enabled_metrics."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        pool, panel = _create_llm_pool_with_judges(
            [("openai", "gpt-4o-mini")],
            enabled_metrics=["ragas:faithfulness", "custom:answer_correctness"],
        )
        system_config = SystemConfig(llm_pool=pool, judge_panel=panel)
        manager = LLMManager.from_system_config(system_config)

        # Metric in list - use panel
        assert manager.should_use_panel_for_metric("ragas:faithfulness")
        assert manager.should_use_panel_for_metric("custom:answer_correctness")

        # Metric not in list - don't use panel
        assert not manager.should_use_panel_for_metric("ragas:response_relevancy")

    def test_should_use_panel_with_enabled_metrics_none(
        self, mocker: MockerFixture
    ) -> None:
        """Test should_use_panel when enabled_metrics is None (all metrics)."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        # enabled_metrics=None is the default, meaning all metrics use panel
        pool, panel = _create_llm_pool_with_judges(
            [("openai", "gpt-4o-mini")],
            enabled_metrics=None,
        )
        system_config = SystemConfig(llm_pool=pool, judge_panel=panel)
        manager = LLMManager.from_system_config(system_config)

        # All metrics use panel
        assert manager.should_use_panel_for_metric("ragas:faithfulness")
        assert manager.should_use_panel_for_metric("custom:answer_correctness")
        assert manager.should_use_panel_for_metric("deepeval:conversation_completeness")

    def test_judge_panel_logs_message(
        self, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test judge panel initialization logs messages."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        pool, panel = _create_llm_pool_with_judges(
            [
                ("openai", "gpt-4o-mini"),
                ("openai", "gpt-4o"),
            ]
        )
        system_config = SystemConfig(llm_pool=pool, judge_panel=panel)

        with caplog.at_level(logging.INFO):
            LLMManager.from_system_config(system_config)

        # Should log judge panel info
        assert any(
            "Judge panel" in record.message and "2 judges" in record.message
            for record in caplog.records
        )


class TestLLMManagerBackwardCompatibility:
    """Tests for backward compatibility with existing llm config (no llm_pool)."""

    def test_from_system_config_without_llm_pool(self, mocker: MockerFixture) -> None:
        """Test that from_system_config works without llm_pool or judge_panel."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        # Create SystemConfig with only llm field (no llm_pool, no judge_panel)
        system_config = SystemConfig(
            llm=LLMConfig(provider="openai", model="gpt-4o-mini")
        )
        manager = LLMManager.from_system_config(system_config)

        # Should work as single-judge mode
        assert not manager.has_judge_panel()
        assert len(manager.judge_managers) == 0
        assert manager.get_primary_judge() is manager
        assert manager.config.model == "gpt-4o-mini"

    def test_get_judge_managers_returns_self_when_no_panel(
        self, mocker: MockerFixture
    ) -> None:
        """Test that get_judge_managers returns [self] when no panel configured."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        system_config = SystemConfig()  # Uses default llm config
        manager = LLMManager.from_system_config(system_config)

        judges = manager.get_judge_managers()
        assert len(judges) == 1
        assert judges[0] is manager

    def test_should_use_panel_returns_false_when_no_panel(
        self, mocker: MockerFixture
    ) -> None:
        """Test that should_use_panel_for_metric returns False without panel."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        system_config = SystemConfig()
        manager = LLMManager.from_system_config(system_config)

        assert not manager.should_use_panel_for_metric("ragas:faithfulness")
        assert not manager.should_use_panel_for_metric("custom:answer_correctness")
