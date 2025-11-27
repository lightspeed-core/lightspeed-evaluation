"""Unit tests for DeepEval LLM Manager."""

import pytest

from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager


@pytest.fixture
def llm_params():
    """Create sample LLM parameters."""
    return {
        "temperature": 0.5,
        "max_tokens": 1024,
        "timeout": 120,
        "num_retries": 5,
    }


class TestDeepEvalLLMManager:
    """Tests for DeepEvalLLMManager."""

    def test_initialization(self, llm_params, mocker):
        """Test manager initialization."""
        mock_model = mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.LiteLLMModel"
        )

        manager = DeepEvalLLMManager("gpt-4", llm_params)

        assert manager.model_name == "gpt-4"
        assert manager.llm_params == llm_params
        mock_model.assert_called_once()

    def test_initialization_with_default_temperature(self, mocker):
        """Test initialization with default temperature."""
        mock_model = mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.LiteLLMModel"
        )

        params = {"max_tokens": 512}
        DeepEvalLLMManager("gpt-3.5-turbo", params)

        # Should use default temperature 0.0
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["temperature"] == 0.0

    def test_initialization_with_default_num_retries(self, mocker):
        """Test initialization with default num_retries."""
        mock_model = mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.LiteLLMModel"
        )

        params = {"temperature": 0.0}
        DeepEvalLLMManager("gpt-4", params)

        # Should use default num_retries 3
        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["num_retries"] == 3

    def test_get_llm(self, llm_params, mocker):
        """Test get_llm method."""
        mock_model_instance = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.LiteLLMModel",
            return_value=mock_model_instance,
        )

        manager = DeepEvalLLMManager("gpt-4", llm_params)
        llm = manager.get_llm()

        assert llm == mock_model_instance

    def test_get_model_info(self, llm_params, mocker):
        """Test get_model_info method."""
        mocker.patch("lightspeed_evaluation.core.llm.deepeval.LiteLLMModel")

        manager = DeepEvalLLMManager("gpt-4", llm_params)
        info = manager.get_model_info()

        assert info["model_name"] == "gpt-4"
        assert info["temperature"] == 0.5
        assert info["max_tokens"] == 1024
        assert info["timeout"] == 120
        assert info["num_retries"] == 5

    def test_initialization_prints_message(self, llm_params, mocker, capsys):
        """Test that initialization prints configuration message."""
        mocker.patch("lightspeed_evaluation.core.llm.deepeval.LiteLLMModel")

        DeepEvalLLMManager("gpt-4", llm_params)

        captured = capsys.readouterr()
        assert "DeepEval LLM Manager" in captured.out
        assert "gpt-4" in captured.out
