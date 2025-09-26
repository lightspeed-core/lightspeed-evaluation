"""Environment variable validation utilities."""

import os

from .exceptions import LLMError


def validate_openai_env() -> None:
    """Validate OpenAI environment variables."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise LLMError(
            "OPENAI_API_KEY environment variable is required for OpenAI provider"
        )


def validate_azure_env() -> None:
    """Validate Azure OpenAI environment variables."""
    required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"]
    if not all(os.environ.get(var) for var in required):
        raise LLMError(f"Azure provider requires environment variables: {required}")


def validate_watsonx_env() -> None:
    """Validate Watsonx environment variables."""
    required = ["WATSONX_API_KEY", "WATSONX_API_BASE", "WATSONX_PROJECT_ID"]
    if not all(os.environ.get(var) for var in required):
        raise LLMError(f"Watsonx provider requires environment variables: {required}")


def validate_anthropic_env() -> None:
    """Validate Anthropic environment variables."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise LLMError(
            "ANTHROPIC_API_KEY environment variable is required for Anthropic provider"
        )


def validate_gemini_env() -> None:
    """Validate Google Gemini environment variables."""
    # Gemini can use either GOOGLE_API_KEY or GEMINI_API_KEY
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        raise LLMError(
            "GOOGLE_API_KEY or GEMINI_API_KEY environment variable "
            "is required for Gemini provider"
        )

def validate_ollama_env() -> None:
    """Validate Ollama environment variables."""
    # Ollama typically runs locally, but may need OLLAMA_HOST for remote instances
    # No required env vars for basic local setup, but warn if OLLAMA_HOST is not set
    if not os.environ.get("OLLAMA_HOST"):
        print("ℹ️ OLLAMA_HOST not set, using default localhost:11434")


def validate_provider_env(provider: str) -> None:
    """Validate environment variables for the specified provider.

    Args:
        provider: The LLM provider name

    Raises:
        LLMError: If required environment variables are missing
    """
    validators = {
        "openai": validate_openai_env,
        "azure": validate_azure_env,
        "watsonx": validate_watsonx_env,
        "anthropic": validate_anthropic_env,
        "gemini": validate_gemini_env,
        "ollama": validate_ollama_env,
    }

    if provider in validators:
        validators[provider]()
