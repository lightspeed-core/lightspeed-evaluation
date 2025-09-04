"""LLM Manager - Generic LLM configuration, validation, and parameter provider."""

import os
from typing import Any, Dict

from ..config import LLMConfig


class LLMError(Exception):
    """LLM configuration error."""


class LLMManager:
    """Generic LLM Manager for all use cases (Ragas, DeepEval, Custom metrics).

    Responsibilities:
    - Environment validation for multiple providers
    - Model name construction
    - Provides LiteLLM parameters for consumption by framework-specific managers
    """

    def __init__(self, config: LLMConfig):
        """Initialize with validated environment and constructed model name."""
        self.config = config
        self.model_name = self._construct_model_name_and_validate()
        print(f"✅ LLM Manager: {self.config.provider}/{self.config.model} -> {self.model_name}")

    def _construct_model_name_and_validate(self) -> str:
        """Construct model name for LiteLLM and validate required environment variables."""
        provider = self.config.provider.lower()

        # Provider-specific validation and model name construction
        provider_handlers = {
            "openai": self._handle_openai_provider,
            "azure": self._handle_azure_provider,
            "watsonx": self._handle_watsonx_provider,
            "anthropic": self._handle_anthropic_provider,
            "gemini": self._handle_gemini_provider,
            "ollama": self._handle_ollama_provider,
        }

        if provider in provider_handlers:
            return provider_handlers[provider]()

        # Generic provider - try as-is with warning
        print(f"⚠️ Using generic provider format for {provider}")
        return f"{provider}/{self.config.model}"

    def _validate_openai_env(self) -> None:
        """Validate OpenAI environment variables."""
        if not os.environ.get("OPENAI_API_KEY"):
            raise LLMError("OPENAI_API_KEY environment variable is required for OpenAI provider")

    def _validate_azure_env(self) -> None:
        """Validate Azure OpenAI environment variables."""
        required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
        if not all(os.environ.get(var) for var in required):
            raise LLMError(f"Azure provider requires environment variables: {required}")

    def _validate_watsonx_env(self) -> None:
        """Validate Watsonx environment variables."""
        required = ["WATSONX_API_KEY", "WATSONX_API_BASE", "WATSONX_PROJECT_ID"]
        if not all(os.environ.get(var) for var in required):
            raise LLMError(f"Watsonx provider requires environment variables: {required}")

    def _validate_anthropic_env(self) -> None:
        """Validate Anthropic environment variables."""
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMError(
                "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
            )

    def _validate_gemini_env(self) -> None:
        """Validate Google Gemini environment variables."""
        # Gemini can use either GOOGLE_API_KEY or GEMINI_API_KEY
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            raise LLMError(
                "GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
                "is required for Gemini provider"
            )

    def _validate_ollama_env(self) -> None:
        """Validate Ollama environment variables."""
        # Ollama typically runs locally, but may need OLLAMA_HOST for remote instances
        # No required env vars for basic local setup, but warn if OLLAMA_HOST is not set
        if not os.environ.get("OLLAMA_HOST"):
            print("ℹ️ OLLAMA_HOST not set, using default localhost:11434")

    def _handle_openai_provider(self) -> str:
        """Handle OpenAI provider setup."""
        self._validate_openai_env()
        return self.config.model

    def _handle_azure_provider(self) -> str:
        """Handle Azure provider setup."""
        self._validate_azure_env()
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or self.config.model
        return f"azure/{deployment}"

    def _handle_watsonx_provider(self) -> str:
        """Handle WatsonX provider setup."""
        self._validate_watsonx_env()
        return f"watsonx/{self.config.model}"

    def _handle_anthropic_provider(self) -> str:
        """Handle Anthropic provider setup."""
        self._validate_anthropic_env()
        return f"anthropic/{self.config.model}"

    def _handle_gemini_provider(self) -> str:
        """Handle Gemini provider setup."""
        self._validate_gemini_env()
        return f"gemini/{self.config.model}"

    def _handle_ollama_provider(self) -> str:
        """Handle Ollama provider setup."""
        self._validate_ollama_env()
        return f"ollama/{self.config.model}"

    def get_model_name(self) -> str:
        """Get the constructed LiteLLM model name."""
        return self.model_name

    def get_litellm_params(self) -> Dict[str, Any]:
        """Get parameters for LiteLLM completion calls."""
        return {
            "model": self.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "num_retries": self.config.num_retries,
        }

    def get_config(self) -> LLMConfig:
        """Get the LLM configuration."""
        return self.config

    @classmethod
    def from_system_config(cls, system_config: Dict[str, Any]) -> "LLMManager":
        """Create LLM Manager from system configuration."""
        llm_config_dict = system_config.get("llm", {})
        config = LLMConfig.from_dict(llm_config_dict)
        return cls(config)
