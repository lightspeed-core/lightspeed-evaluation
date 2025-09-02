"""Unit tests for core.llm.manager module."""

import os
from unittest.mock import patch

import pytest

from lightspeed_evaluation.core.config.models import LLMConfig
from lightspeed_evaluation.core.llm.manager import LLMError, LLMManager


class TestLLMError:
    """Unit tests for LLMError exception."""

    def test_llm_error_creation(self):
        """Test creating LLMError exception."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"

    def test_llm_error_inheritance(self):
        """Test that LLMError inherits from Exception."""
        error = LLMError("Test error")
        assert isinstance(error, Exception)


class TestLLMManager:
    """Unit tests for LLMManager class."""

    def test_llm_manager_initialization_openai(self):
        """Test LLMManager initialization with OpenAI provider."""
        config = LLMConfig(provider="openai", model="gpt-4")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("builtins.print") as mock_print:
                manager = LLMManager(config)

                assert manager.config == config
                assert manager.model_name == "gpt-4"
                mock_print.assert_called_with("✅ LLM Manager: openai/gpt-4 -> gpt-4")

    def test_llm_manager_initialization_anthropic(self):
        """Test LLMManager initialization with Anthropic provider."""
        config = LLMConfig(provider="anthropic", model="claude-3-sonnet")

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch("builtins.print") as mock_print:
                manager = LLMManager(config)

                assert manager.config == config
                assert manager.model_name == "anthropic/claude-3-sonnet"
                mock_print.assert_called_with(
                    "✅ LLM Manager: anthropic/claude-3-sonnet -> anthropic/claude-3-sonnet"
                )

    def test_llm_manager_initialization_azure(self):
        """Test LLMManager initialization with Azure provider."""
        config = LLMConfig(provider="azure", model="gpt-4")

        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
            },
        ):
            with patch("builtins.print") as mock_print:
                manager = LLMManager(config)

                assert manager.config == config
                assert manager.model_name == "azure/gpt-4"

    def test_llm_manager_initialization_azure_with_deployment(self):
        """Test LLMManager initialization with Azure provider and custom deployment."""
        config = LLMConfig(provider="azure", model="gpt-4")

        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_API_KEY": "test-key",
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
                "AZURE_OPENAI_DEPLOYMENT_NAME": "custom-deployment",
            },
        ):
            manager = LLMManager(config)
            assert manager.model_name == "azure/custom-deployment"

    def test_llm_manager_initialization_watsonx(self):
        """Test LLMManager initialization with Watsonx provider."""
        config = LLMConfig(provider="watsonx", model="llama-2-70b")

        with patch.dict(
            os.environ,
            {
                "WATSONX_API_KEY": "test-key",
                "WATSONX_API_BASE": "https://test.watsonx.ai",
                "WATSONX_PROJECT_ID": "test-project",
            },
        ):
            manager = LLMManager(config)
            assert manager.model_name == "watsonx/llama-2-70b"

    def test_llm_manager_initialization_gemini_google_key(self):
        """Test LLMManager initialization with Gemini provider using GOOGLE_API_KEY."""
        config = LLMConfig(provider="gemini", model="gemini-pro")

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            manager = LLMManager(config)
            assert manager.model_name == "gemini/gemini-pro"

    def test_llm_manager_initialization_gemini_gemini_key(self):
        """Test LLMManager initialization with Gemini provider using GEMINI_API_KEY."""
        config = LLMConfig(provider="gemini", model="gemini-pro")

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            manager = LLMManager(config)
            assert manager.model_name == "gemini/gemini-pro"

    def test_llm_manager_initialization_ollama(self):
        """Test LLMManager initialization with Ollama provider."""
        config = LLMConfig(provider="ollama", model="llama2")

        with patch("builtins.print") as mock_print:
            manager = LLMManager(config)

            assert manager.model_name == "ollama/llama2"
            mock_print.assert_any_call(
                "ℹ️ OLLAMA_HOST not set, using default localhost:11434"
            )

    def test_llm_manager_initialization_ollama_with_host(self):
        """Test LLMManager initialization with Ollama provider and custom host."""
        config = LLMConfig(provider="ollama", model="llama2")

        with patch.dict(os.environ, {"OLLAMA_HOST": "http://custom-host:11434"}):
            with patch("builtins.print") as mock_print:
                manager = LLMManager(config)

                assert manager.model_name == "ollama/llama2"
                # Should not print the warning about missing OLLAMA_HOST
                for call in mock_print.call_args_list:
                    assert "OLLAMA_HOST not set" not in str(call)

    def test_llm_manager_initialization_generic_provider(self):
        """Test LLMManager initialization with unknown/generic provider."""
        config = LLMConfig(provider="custom", model="custom-model")

        with patch("builtins.print") as mock_print:
            manager = LLMManager(config)

            assert manager.model_name == "custom/custom-model"
            mock_print.assert_any_call("⚠️ Using generic provider format for custom")

    def test_llm_manager_openai_missing_api_key(self):
        """Test LLMManager with OpenAI provider but missing API key."""
        config = LLMConfig(provider="openai", model="gpt-4")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                LLMError, match="OPENAI_API_KEY environment variable is required"
            ):
                LLMManager(config)

    def test_llm_manager_azure_missing_api_key(self):
        """Test LLMManager with Azure provider but missing API key."""
        config = LLMConfig(provider="azure", model="gpt-4")

        with patch.dict(
            os.environ, {"AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"}
        ):
            with pytest.raises(
                LLMError, match="Azure provider requires environment variables"
            ):
                LLMManager(config)

    def test_llm_manager_azure_missing_endpoint(self):
        """Test LLMManager with Azure provider but missing endpoint."""
        config = LLMConfig(provider="azure", model="gpt-4")

        with patch.dict(os.environ, {"AZURE_OPENAI_API_KEY": "test-key"}):
            with pytest.raises(
                LLMError, match="Azure provider requires environment variables"
            ):
                LLMManager(config)

    def test_llm_manager_watsonx_missing_api_key(self):
        """Test LLMManager with Watsonx provider but missing API key."""
        config = LLMConfig(provider="watsonx", model="llama-2-70b")

        with patch.dict(
            os.environ,
            {
                "WATSONX_API_BASE": "https://test.watsonx.ai",
                "WATSONX_PROJECT_ID": "test-project",
            },
        ):
            with pytest.raises(
                LLMError, match="Watsonx provider requires environment variables"
            ):
                LLMManager(config)

    def test_llm_manager_anthropic_missing_api_key(self):
        """Test LLMManager with Anthropic provider but missing API key."""
        config = LLMConfig(provider="anthropic", model="claude-3-sonnet")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                LLMError, match="ANTHROPIC_API_KEY environment variable is required"
            ):
                LLMManager(config)

    def test_llm_manager_gemini_missing_api_key(self):
        """Test LLMManager with Gemini provider but missing API key."""
        config = LLMConfig(provider="gemini", model="gemini-pro")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                LLMError, match="GOOGLE_API_KEY or GEMINI_API_KEY environment variable"
            ):
                LLMManager(config)

    def test_get_model_name(self):
        """Test get_model_name method."""
        config = LLMConfig(provider="openai", model="gpt-4")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config)
            assert manager.get_model_name() == "gpt-4"

    def test_get_litellm_params(self):
        """Test get_litellm_params method."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            num_retries=3,
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config)
            params = manager.get_litellm_params()

            expected = {
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 60,
                "num_retries": 3,
            }
            assert params == expected

    def test_get_config(self):
        """Test get_config method."""
        config = LLMConfig(provider="openai", model="gpt-4")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config)
            assert manager.get_config() == config

    def test_from_system_config(self):
        """Test from_system_config class method."""
        system_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.5,
                "max_tokens": 2000,
            }
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager.from_system_config(system_config)

            assert manager.config.provider == "openai"
            assert manager.config.model == "gpt-4"
            assert manager.config.temperature == 0.5
            assert manager.config.max_tokens == 2000

    def test_from_system_config_missing_llm_section(self):
        """Test from_system_config with missing llm section."""
        system_config = {}

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager.from_system_config(system_config)

            # Should use defaults from LLMConfig
            assert manager.config.provider == "openai"  # default
            assert manager.config.model == "gpt-4o-mini"  # default

    def test_provider_case_insensitive(self):
        """Test that provider names are handled case-insensitively."""
        config = LLMConfig(provider="OpenAI", model="gpt-4")

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config)
            assert manager.model_name == "gpt-4"

    def test_multiple_providers_in_sequence(self):
        """Test creating managers for different providers in sequence."""
        providers_data = [
            ("openai", "gpt-4", {"OPENAI_API_KEY": "test-key"}, "gpt-4"),
            (
                "anthropic",
                "claude-3-sonnet",
                {"ANTHROPIC_API_KEY": "test-key"},
                "anthropic/claude-3-sonnet",
            ),
            (
                "gemini",
                "gemini-pro",
                {"GOOGLE_API_KEY": "test-key"},
                "gemini/gemini-pro",
            ),
        ]

        for provider, model, env_vars, expected_model_name in providers_data:
            config = LLMConfig(provider=provider, model=model)

            with patch.dict(os.environ, env_vars):
                with patch("builtins.print"):
                    manager = LLMManager(config)
                    assert manager.model_name == expected_model_name
                    assert manager.config.provider == provider
                    assert manager.config.model == model
