"""LLM Manager - Generic LLM configuration, validation, and parameter provider."""

import os
from typing import Any

from ..models import LLMConfig, SystemConfig
from ..system.env_validator import validate_provider_env


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
        print(
            f"✅ LLM Manager: {self.config.provider}/{self.config.model} -> {self.model_name}"
        )

    def _construct_model_name_and_validate(self) -> str:
        """Construct model name for LiteLLM and validate required environment variables."""
        provider = self.config.provider.lower()

        # Provider-specific validation and model name construction
        provider_handlers = {
            "hosted_vllm": self._handle_hosted_vllm_provider,
            "openai": self._handle_openai_provider,
            "azure": self._handle_azure_provider,
            "watsonx": self._handle_watsonx_provider,
            "anthropic": self._handle_anthropic_provider,
            "gemini": self._handle_gemini_provider,
            "vertex": self._handle_vertex_provider,
            "ollama": self._handle_ollama_provider,
        }

        if provider in provider_handlers:
            return provider_handlers[provider]()

        # Generic provider - try as-is with warning
        print(f"⚠️ Using generic provider format for {provider}")
        return f"{provider}/{self.config.model}"

    def _handle_hosted_vllm_provider(self) -> str:
        """Handle hosted vLLM provider setup."""
        validate_provider_env("hosted_vllm")
        return f"hosted_vllm/{self.config.model}"

    def _handle_openai_provider(self) -> str:
        """Handle OpenAI provider setup."""
        validate_provider_env("openai")
        return self.config.model

    def _handle_azure_provider(self) -> str:
        """Handle Azure provider setup."""
        validate_provider_env("azure")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME") or self.config.model
        return f"azure/{deployment}"

    def _handle_watsonx_provider(self) -> str:
        """Handle WatsonX provider setup."""
        validate_provider_env("watsonx")
        return f"watsonx/{self.config.model}"

    def _handle_anthropic_provider(self) -> str:
        """Handle Anthropic provider setup."""
        validate_provider_env("anthropic")
        return f"anthropic/{self.config.model}"

    def _handle_gemini_provider(self) -> str:
        """Handle Gemini provider setup."""
        validate_provider_env("gemini")
        return f"gemini/{self.config.model}"

    def _handle_vertex_provider(self) -> str:
        """Handle Vertex AI provider setup."""
        validate_provider_env("vertex")
        return self.config.model

    def _handle_ollama_provider(self) -> str:
        """Handle Ollama provider setup."""
        validate_provider_env("ollama")
        return f"ollama/{self.config.model}"

    def get_model_name(self) -> str:
        """Get the constructed LiteLLM model name."""
        return self.model_name

    def get_litellm_params(self) -> dict[str, Any]:
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
    def from_system_config(cls, system_config: SystemConfig) -> "LLMManager":
        """Create LLM Manager from system configuration."""
        return cls(system_config.llm)

    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig) -> "LLMManager":
        """Create LLM Manager from LLMConfig directly."""
        return cls(llm_config)
