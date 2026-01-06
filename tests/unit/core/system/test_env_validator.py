"""Unit tests for environment validator."""

import pytest

from lightspeed_evaluation.core.system.env_validator import (
    validate_anthropic_env,
    validate_azure_env,
    validate_gemini_env,
    validate_hosted_vllm_env,
    validate_ollama_env,
    validate_openai_env,
    validate_provider_env,
    validate_vertex_env,
    validate_watsonx_env,
)
from lightspeed_evaluation.core.system.exceptions import LLMError


class TestProviderValidators:
    """Tests for individual provider validators."""

    def test_validate_openai_env_success(self, mocker):
        """Test OpenAI validation succeeds with API key."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})

        # Should not raise
        validate_openai_env()

    def test_validate_openai_env_failure(self, mocker):
        """Test OpenAI validation fails without API key."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(LLMError, match="OPENAI_API_KEY"):
            validate_openai_env()

    def test_validate_azure_env_success(self, mocker):
        """Test Azure validation succeeds with required vars."""
        mocker.patch.dict(
            "os.environ",
            {
                "AZURE_API_KEY": "test_key",
                "AZURE_API_BASE": "https://test.openai.azure.com/",
            },
        )

        validate_azure_env()

    def test_validate_azure_env_failure(self, mocker):
        """Test Azure validation fails without required vars."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(LLMError, match="Azure"):
            validate_azure_env()

    def test_validate_watsonx_env_success(self, mocker):
        """Test Watsonx validation succeeds with required vars."""
        mocker.patch.dict(
            "os.environ",
            {
                "WATSONX_API_KEY": "test_key",
                "WATSONX_API_BASE": "https://test.watsonx.com",
                "WATSONX_PROJECT_ID": "proj_123",
            },
        )

        validate_watsonx_env()

    def test_validate_watsonx_env_failure(self, mocker):
        """Test Watsonx validation fails without required vars."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(LLMError, match="Watsonx"):
            validate_watsonx_env()

    def test_validate_anthropic_env_success(self, mocker):
        """Test Anthropic validation succeeds with API key."""
        mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"})

        validate_anthropic_env()

    def test_validate_anthropic_env_failure(self, mocker):
        """Test Anthropic validation fails without API key."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
            validate_anthropic_env()

    def test_validate_gemini_env_with_google_api_key(self, mocker):
        """Test Gemini validation succeeds with GOOGLE_API_KEY."""
        mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "test_key"})

        validate_gemini_env()

    def test_validate_gemini_env_with_gemini_api_key(self, mocker):
        """Test Gemini validation succeeds with GEMINI_API_KEY."""
        mocker.patch.dict("os.environ", {"GEMINI_API_KEY": "test_key"})

        validate_gemini_env()

    def test_validate_gemini_env_failure(self, mocker):
        """Test Gemini validation fails without API keys."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(LLMError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
            validate_gemini_env()

    def test_validate_vertex_env_success(self, mocker):
        """Test Vertex AI validation succeeds with credentials."""
        mocker.patch.dict(
            "os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds.json"}
        )

        validate_vertex_env()

    def test_validate_vertex_env_failure(self, mocker):
        """Test Vertex AI validation fails without credentials."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(LLMError, match="GOOGLE_APPLICATION_CREDENTIALS"):
            validate_vertex_env()

    def test_validate_ollama_env_with_host(self, mocker):
        """Test Ollama validation with OLLAMA_HOST set."""
        mocker.patch.dict("os.environ", {"OLLAMA_HOST": "http://localhost:11434"})

        # Should not raise or print warning
        validate_ollama_env()

    def test_validate_ollama_env_without_host(self, mocker, capsys):
        """Test Ollama validation without OLLAMA_HOST prints info."""
        mocker.patch.dict("os.environ", {}, clear=True)

        validate_ollama_env()

        captured = capsys.readouterr()
        assert "OLLAMA_HOST" in captured.out or "localhost" in captured.out

    def test_validate_hosted_vllm_env_success(self, mocker):
        """Test hosted vLLM validation succeeds with required vars."""
        mocker.patch.dict(
            "os.environ",
            {
                "HOSTED_VLLM_API_KEY": "test_key",
                "HOSTED_VLLM_API_BASE": "https://vllm.host.com",
            },
        )

        validate_hosted_vllm_env()

    def test_validate_hosted_vllm_env_failure(self, mocker):
        """Test hosted vLLM validation fails without required vars."""
        mocker.patch.dict("os.environ", {}, clear=True)

        with pytest.raises(LLMError, match="Hosted vLLM"):
            validate_hosted_vllm_env()


class TestValidateProviderEnv:
    """Tests for validate_provider_env dispatcher."""

    def test_validate_provider_openai(self, mocker):
        """Test provider validation dispatches to OpenAI validator."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test"})

        validate_provider_env("openai")

    def test_validate_provider_azure(self, mocker):
        """Test provider validation dispatches to Azure validator."""
        mocker.patch.dict(
            "os.environ",
            {
                "AZURE_API_KEY": "test",
                "AZURE_API_BASE": "https://test.com",
            },
        )

        validate_provider_env("azure")

    def test_validate_provider_watsonx(self, mocker):
        """Test provider validation dispatches to Watsonx validator."""
        mocker.patch.dict(
            "os.environ",
            {
                "WATSONX_API_KEY": "test",
                "WATSONX_API_BASE": "https://test.com",
                "WATSONX_PROJECT_ID": "proj",
            },
        )

        validate_provider_env("watsonx")

    def test_validate_provider_unknown(self, mocker):
        """Test unknown provider doesn't raise error."""
        # Unknown providers should be handled gracefully
        validate_provider_env("unknown_provider")

    def test_validate_provider_anthropic(self, mocker):
        """Test provider validation for Anthropic."""
        mocker.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"})

        validate_provider_env("anthropic")

    def test_validate_provider_gemini(self, mocker):
        """Test provider validation for Gemini."""
        mocker.patch.dict("os.environ", {"GOOGLE_API_KEY": "test"})

        validate_provider_env("gemini")

    def test_validate_provider_vertex(self, mocker):
        """Test provider validation for Vertex AI."""
        mocker.patch.dict(
            "os.environ", {"GOOGLE_APPLICATION_CREDENTIALS": "/path/to/creds"}
        )

        validate_provider_env("vertex")

    def test_validate_provider_ollama(self, mocker):
        """Test provider validation for Ollama."""
        mocker.patch.dict("os.environ", {})

        validate_provider_env("ollama")

    def test_validate_provider_hosted_vllm(self, mocker):
        """Test provider validation for hosted vLLM."""
        mocker.patch.dict(
            "os.environ",
            {"HOSTED_VLLM_API_KEY": "test", "HOSTED_VLLM_API_BASE": "https://test.com"},
        )

        validate_provider_env("hosted_vllm")
