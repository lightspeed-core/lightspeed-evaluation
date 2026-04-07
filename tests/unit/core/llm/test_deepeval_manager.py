"""Unit tests for DeepEval LLM Manager."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager


class TestDeepEvalLLMManager:
    """Tests for DeepEvalLLMManager."""

    def test_setup_ssl_verify_delegates_to_litellm_patch(
        self, mocker: MockerFixture
    ) -> None:
        """Test SSL verification delegates to setup_litellm_ssl."""
        mock_setup = mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.setup_litellm_ssl"
        )
        mocker.patch("lightspeed_evaluation.core.llm.deepeval.LiteLLMModel")

        params = {"ssl_verify": False}
        DeepEvalLLMManager("gpt-4", params)

        mock_setup.assert_called_once_with(params)

    def test_initialization(self, llm_params: dict, mocker: MockerFixture) -> None:
        """Test manager initialization."""
        mock_model = mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.LiteLLMModel"
        )

        manager = DeepEvalLLMManager("gpt-4", llm_params)

        assert manager.model_name == "gpt-4"
        assert manager.llm_params == llm_params
        mock_model.assert_called_once()

    def test_initialization_passes_parameters(self, mocker: MockerFixture) -> None:
        """Test initialization passes parameters to LiteLLMModel."""
        mock_model = mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.LiteLLMModel"
        )

        params = {"parameters": {"temperature": 0.7, "max_completion_tokens": 512}}
        DeepEvalLLMManager("gpt-3.5-turbo", params)

        call_kwargs = mock_model.call_args.kwargs
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_completion_tokens"] == 512

    def test_get_llm(self, llm_params: dict, mocker: MockerFixture) -> None:
        """Test get_llm method."""
        mock_model_instance = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.llm.deepeval.LiteLLMModel",
            return_value=mock_model_instance,
        )

        manager = DeepEvalLLMManager("gpt-4", llm_params)
        llm = manager.get_llm()

        assert llm == mock_model_instance

    def test_get_model_info(self, llm_params: dict, mocker: MockerFixture) -> None:
        """Test get_model_info method."""
        mocker.patch("lightspeed_evaluation.core.llm.deepeval.LiteLLMModel")

        manager = DeepEvalLLMManager("gpt-4", llm_params)
        info = manager.get_model_info()

        assert info["model_name"] == "gpt-4"
        assert info["timeout"] == 120
        assert info["num_retries"] == 5
        # Parameters from nested dict
        assert info["temperature"] == 0.5
        assert info["max_completion_tokens"] == 1024

    def test_initialization_prints_message(
        self, llm_params: dict, mocker: MockerFixture, capsys: pytest.CaptureFixture
    ) -> None:
        """Test that initialization prints configuration message."""
        mocker.patch("lightspeed_evaluation.core.llm.deepeval.LiteLLMModel")

        DeepEvalLLMManager("gpt-4", llm_params)

        captured = capsys.readouterr()
        assert "DeepEval LLM Manager" in captured.out
        assert "gpt-4" in captured.out

    def test_drop_params_always_enabled(self, mocker: MockerFixture) -> None:
        """Test drop_params is always enabled for cross-provider compatibility."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.deepeval.litellm")
        mocker.patch("lightspeed_evaluation.core.llm.deepeval.LiteLLMModel")

        DeepEvalLLMManager("gpt-4", {})

        assert mock_litellm.drop_params is True
