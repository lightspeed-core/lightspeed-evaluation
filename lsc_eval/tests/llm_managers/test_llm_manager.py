"""Test cases for LLM Manager based on system.yaml configuration."""

import os
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import pytest

from lsc_eval.llm_managers.llm_manager import LLMManager, LLMConfig, LLMError


class TestLLMConfig:
    """Test LLMConfig dataclass."""

    def test_llm_config_creation_with_defaults(self):
        """Test LLMConfig creation with default values."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")

        assert config.provider == "openai"
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 512
        assert config.timeout == 300
        assert config.num_retries == 3

    def test_llm_config_creation_with_custom_values(self):
        """Test LLMConfig creation with custom values."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-sonnet",
            temperature=0.5,
            max_tokens=1024,
            timeout=600,
            num_retries=5
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-sonnet"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024
        assert config.timeout == 600
        assert config.num_retries == 5

    def test_llm_config_from_dict_with_all_values(self):
        """Test LLMConfig.from_dict with all values provided."""
        config_dict = {
            "provider": "azure",
            "model": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 2048,
            "timeout": 450,
            "num_retries": 2
        }

        config = LLMConfig.from_dict(config_dict)

        assert config.provider == "azure"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2048
        assert config.timeout == 450
        assert config.num_retries == 2

    def test_llm_config_from_dict_with_defaults(self):
        """Test LLMConfig.from_dict with missing values using defaults."""
        config_dict = {
            "provider": "watsonx",
            "model": "llama-2-70b"
        }

        config = LLMConfig.from_dict(config_dict)

        assert config.provider == "watsonx"
        assert config.model == "llama-2-70b"
        assert config.temperature == 0.0  # Default
        assert config.max_tokens == 512   # Default
        assert config.timeout == 300      # Default
        assert config.num_retries == 3    # Default

    def test_llm_config_from_dict_empty_dict(self):
        """Test LLMConfig.from_dict with empty dictionary."""
        config = LLMConfig.from_dict({})

        assert config.provider == "openai"     # Default
        assert config.model == "gpt-4o-mini"   # Default
        assert config.temperature == 0.0
        assert config.max_tokens == 512
        assert config.timeout == 300
        assert config.num_retries == 3


class TestLLMManager:
    """Test LLMManager class functionality."""

    def test_llm_manager_provider_initialization(self):
        """Test LLMManager initialization with different providers."""
        test_cases = [
            ("openai", "gpt-4o-mini", {"OPENAI_API_KEY": "key"}, "gpt-4o-mini"),
            ("azure", "gpt-4", {"AZURE_OPENAI_API_KEY": "key", "AZURE_OPENAI_ENDPOINT": "endpoint"}, "azure/gpt-4"),
            ("anthropic", "claude-3-sonnet", {"ANTHROPIC_API_KEY": "key"}, "anthropic/claude-3-sonnet"),
            ("gemini", "gemini-pro", {"GOOGLE_API_KEY": "key"}, "gemini/gemini-pro"),
            ("watsonx", "llama-2", {"WATSONX_API_KEY": "key", "WATSONX_API_BASE": "base", "WATSONX_PROJECT_ID": "proj"}, "watsonx/llama-2"),
            ("ollama", "llama2", {}, "ollama/llama2"),
            ("custom", "model", {}, "custom/model"),
        ]

        for provider, model, env_vars, expected_model_name in test_cases:
            config = LLMConfig(provider=provider, model=model)
            
            with patch.dict(os.environ, env_vars, clear=True):
                with patch('builtins.print'):
                    manager = LLMManager(config)

            assert manager.model_name == expected_model_name

    def test_llm_manager_missing_credentials(self):
        """Test LLMManager with missing credentials for various providers."""
        error_cases = [
            ("openai", "gpt-4o-mini", {}, "OPENAI_API_KEY"),
            ("azure", "gpt-4", {"AZURE_OPENAI_ENDPOINT": "test"}, "Azure provider requires"),
            ("anthropic", "claude-3-sonnet", {}, "ANTHROPIC_API_KEY"),
            ("gemini", "gemini-pro", {}, "GOOGLE_API_KEY or GEMINI_API_KEY"),
            ("watsonx", "llama-2", {}, "Watsonx provider requires"),
        ]

        for provider, model, env_vars, expected_error in error_cases:
            config = LLMConfig(provider=provider, model=model)
            
            with patch.dict(os.environ, env_vars, clear=True):
                with pytest.raises(LLMError, match=expected_error):
                    LLMManager(config)

    def test_get_model_name(self):
        """Test get_model_name method."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager(config)

        assert manager.get_model_name() == "gpt-4o-mini"

    def test_get_litellm_params(self):
        """Test get_litellm_params method."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            timeout=600,
            num_retries=5
        )
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager(config)

        params = manager.get_litellm_params()
        
        expected_params = {
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1024,
            "timeout": 600,
            "num_retries": 5
        }
        
        assert params == expected_params

    def test_get_config(self):
        """Test get_config method."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager(config)

        assert manager.get_config() == config

    def test_from_system_config_full_config(self):
        """Test from_system_config with full configuration."""
        system_config = {
            "llm": {
                "provider": "anthropic",
                "model": "claude-3-haiku",
                "temperature": 0.3,
                "max_tokens": 2048,
                "timeout": 450,
                "num_retries": 2
            }
        }
        
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager.from_system_config(system_config)

        assert manager.config.provider == "anthropic"
        assert manager.config.model == "claude-3-haiku"
        assert manager.config.temperature == 0.3
        assert manager.config.max_tokens == 2048
        assert manager.config.timeout == 450
        assert manager.config.num_retries == 2

    def test_from_system_config_minimal_config(self):
        """Test from_system_config with minimal configuration."""
        system_config = {
            "llm": {
                "provider": "openai",
                "model": "gpt-4o-mini"
            }
        }
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager.from_system_config(system_config)

        assert manager.config.provider == "openai"
        assert manager.config.model == "gpt-4o-mini"
        assert manager.config.temperature == 0.0  # Default
        assert manager.config.max_tokens == 512   # Default

    def test_from_system_config_empty_llm_section(self):
        """Test from_system_config with empty LLM section."""
        system_config = {"llm": {}}
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager.from_system_config(system_config)

        assert manager.config.provider == "openai"     # Default
        assert manager.config.model == "gpt-4o-mini"   # Default

    def test_from_system_config_missing_llm_section(self):
        """Test from_system_config with missing LLM section."""
        system_config = {"other_section": {}}
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager.from_system_config(system_config)

        assert manager.config.provider == "openai"     # Default
        assert manager.config.model == "gpt-4o-mini"   # Default

    @patch('builtins.print')
    def test_print_statements_successful_init(self, mock_print):
        """Test print statements during successful initialization."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            manager = LLMManager(config)

        mock_print.assert_called_with("✅ LLM Manager: openai/gpt-4o-mini -> gpt-4o-mini")

    @patch('builtins.print')
    def test_print_statements_generic_provider(self, mock_print):
        """Test print statements for generic provider."""
        config = LLMConfig(provider="unknown_provider", model="custom-model")
        
        manager = LLMManager(config)

        # Should print warning for generic provider
        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("⚠️ Using generic provider format for unknown_provider" in call for call in print_calls)
        assert any("✅ LLM Manager: unknown_provider/custom-model -> unknown_provider/custom-model" in call for call in print_calls)

    @patch('builtins.print')
    def test_print_statements_ollama_no_host(self, mock_print):
        """Test print statements for Ollama without OLLAMA_HOST."""
        config = LLMConfig(provider="ollama", model="llama2")
        
        with patch.dict(os.environ, {}, clear=True):
            manager = LLMManager(config)

        print_calls = [call.args[0] for call in mock_print.call_args_list]
        assert any("ℹ️ OLLAMA_HOST not set, using default localhost:11434" in call for call in print_calls)

    def test_case_insensitive_provider_handling(self):
        """Test that provider names are handled case-insensitively."""
        config = LLMConfig(provider="OpenAI", model="gpt-4o-mini")  # Mixed case
        
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch('builtins.print'):
                manager = LLMManager(config)

        assert manager.model_name == "gpt-4o-mini"  # Should work despite mixed case

    def test_provider_specific_model_name_construction(self):
        """Test that different providers construct model names correctly."""
        test_cases = [
            ("openai", "gpt-4o-mini", {"OPENAI_API_KEY": "key"}, "gpt-4o-mini"),
            ("azure", "gpt-4", {"AZURE_OPENAI_API_KEY": "key", "AZURE_OPENAI_ENDPOINT": "endpoint"}, "azure/gpt-4"),
            ("anthropic", "claude-3-sonnet", {"ANTHROPIC_API_KEY": "key"}, "anthropic/claude-3-sonnet"),
            ("gemini", "gemini-pro", {"GOOGLE_API_KEY": "key"}, "gemini/gemini-pro"),
            ("watsonx", "llama-2", {"WATSONX_API_KEY": "key", "WATSONX_API_BASE": "base", "WATSONX_PROJECT_ID": "proj"}, "watsonx/llama-2"),
            ("ollama", "llama2", {}, "ollama/llama2"),
        ]

        for provider, model, env_vars, expected_model_name in test_cases:
            config = LLMConfig(provider=provider, model=model)
            
            with patch.dict(os.environ, env_vars, clear=True):
                with patch('builtins.print'):
                    manager = LLMManager(config)

            assert manager.model_name == expected_model_name, f"Failed for provider {provider}"


class TestLLMError:
    """Test LLMError exception."""

    def test_llm_error_creation(self):
        """Test LLMError exception creation."""
        error = LLMError("Test error message")
        assert str(error) == "Test error message"

    def test_llm_error_inheritance(self):
        """Test that LLMError inherits from Exception."""
        error = LLMError("Test error")
        assert isinstance(error, Exception)

