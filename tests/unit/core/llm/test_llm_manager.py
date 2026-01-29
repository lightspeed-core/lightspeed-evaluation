"""Unit tests for LLM Manager."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import LLMConfig, SystemConfig
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
        self, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test initialization with unknown/generic provider."""
        config = LLMConfig(
            provider="custom_provider",
            model="custom-model",
            temperature=0.0,
        )
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        manager = LLMManager(config)

        # Should construct generic model name
        assert manager.model_name == "custom_provider/custom-model"

        # Should print warning
        captured = capsys.readouterr()
        assert "generic" in captured.out.lower() or "warning" in captured.out.lower()

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

    def test_initialization_prints_message(
        self,
        basic_llm_config: LLMConfig,
        mocker: MockerFixture,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Test that initialization prints configuration message."""
        mocker.patch("lightspeed_evaluation.core.llm.manager.validate_provider_env")

        LLMManager(basic_llm_config)

        captured = capsys.readouterr()
        assert "LLM Manager" in captured.out
        assert "openai" in captured.out
        assert "gpt-4" in captured.out
