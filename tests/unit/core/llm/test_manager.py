"""Unit tests for core.llm.manager module."""

import os
from pytest_mock import MockerFixture

import pytest
from lightspeed_evaluation.core.llm import LLMError
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.models import LLMConfig, SystemConfig


class TestLLMError:
    """Unit tests for LLMError exception."""

    def test_llm_error_creation(self) -> None:
        """Test creating LLMError exception."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"

    def test_llm_error_inheritance(self) -> None:
        """Test that LLMError inherits from Exception."""
        error = LLMError("Test error")
        assert isinstance(error, Exception)


class TestLLMManager:
    """Unit tests for LLMManager class."""

    def test_llm_manager_initialization_openai(self, mocker: MockerFixture) -> None:
        """Test LLMManager initialization with OpenAI provider."""
        config = LLMConfig(provider="openai", model="gpt-4")

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        mock_print = mocker.patch("builtins.print")
        manager = LLMManager(config)

        assert manager.config == config
        assert manager.model_name == "gpt-4"
        mock_print.assert_called_with("✅ LLM Manager: openai/gpt-4 -> gpt-4")

    def test_llm_manager_initialization_generic_provider(
        self, mocker: MockerFixture
    ) -> None:
        """Test LLMManager initialization with unknown/generic provider."""
        config = LLMConfig(provider="custom", model="custom-model")

        mock_print = mocker.patch("builtins.print")
        manager = LLMManager(config)

        assert manager.model_name == "custom/custom-model"
        mock_print.assert_any_call("⚠️ Using generic provider format for custom")

    def test_llm_manager_openai_missing_api_key(self, mocker: MockerFixture) -> None:
        """Test LLMManager with OpenAI provider but missing API key."""
        config = LLMConfig(provider="openai", model="gpt-4")

        mocker.patch.dict(os.environ, {}, clear=True)
        with pytest.raises(
            LLMError, match="OPENAI_API_KEY environment variable is required"
        ):
            LLMManager(config)

    def test_get_model_name(self, mocker: MockerFixture) -> None:
        """Test get_model_name method."""
        config = LLMConfig(provider="openai", model="gpt-4")

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        manager = LLMManager(config)
        assert manager.get_model_name() == "gpt-4"

    def test_get_llm_params(self, mocker: MockerFixture) -> None:
        """Test get_llm_params method."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000,
            timeout=60,
            num_retries=3,
        )

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        manager = LLMManager(config)
        params = manager.get_llm_params()

        expected = {
            "model": "gpt-4",
            "temperature": 0.7,
            "max_completion_tokens": 1000,
            "timeout": 60,
            "num_retries": 3,
            "ssl_verify": True,
        }
        assert params == expected

    def test_get_llm_params_with_ssl_verify_false(self, mocker: MockerFixture) -> None:
        """Test get_llm_params method with ssl_verify set to False."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.5,
            max_tokens=512,
            timeout=30,
            num_retries=2,
            ssl_verify=False,
        )

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        manager = LLMManager(config)
        params = manager.get_llm_params()

        expected = {
            "model": "gpt-4",
            "temperature": 0.5,
            "max_completion_tokens": 512,
            "timeout": 30,
            "num_retries": 2,
            "ssl_verify": False,
        }
        assert params == expected

    def test_get_config(self, mocker: MockerFixture) -> None:
        """Test get_config method."""
        config = LLMConfig(provider="openai", model="gpt-4")

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        manager = LLMManager(config)
        assert manager.get_config() == config

    def test_from_system_config(self, mocker: MockerFixture) -> None:
        """Test from_system_config class method."""
        system_config = SystemConfig.model_validate(
            {
                "llm": {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "max_tokens": 2000,
                },
            }
        )

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        manager = LLMManager.from_system_config(system_config)

        assert manager.config.provider == "openai"
        assert manager.config.model == "gpt-4"
        assert manager.config.temperature == 0.5
        assert manager.config.max_tokens == 2000

    def test_provider_case_insensitive(self, mocker: MockerFixture) -> None:
        """Test that provider names are handled case-insensitively."""
        config = LLMConfig(provider="OpenAI", model="gpt-4")

        mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
        manager = LLMManager(config)
        assert manager.model_name == "gpt-4"

    def test_multiple_providers_in_sequence(self, mocker: MockerFixture) -> None:
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

            mocker.patch.dict(os.environ, env_vars)
            mocker.patch("builtins.print")
            manager = LLMManager(config)
            assert manager.model_name == expected_model_name
            assert manager.config.provider == provider
            assert manager.config.model == model
