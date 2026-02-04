# pylint: disable=protected-access,disable=too-few-public-methods

"""Unit tests for custom LLM classes."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.llm.custom import BaseCustomLLM, TokenTracker
from lightspeed_evaluation.core.system.exceptions import LLMError


class TestTokenTracker:
    """Tests for TokenTracker."""

    def test_token_callback_accumulates_tokens(self, mocker: MockerFixture) -> None:
        """Test that token callback accumulates token counts."""
        tracker = TokenTracker()

        # Mock completion response with usage
        mock_response = mocker.Mock()
        mock_response.usage = mocker.Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20

        tracker._token_callback({}, mock_response, 0.0, 0.0)

        input_tokens, output_tokens = tracker.get_counts()
        assert input_tokens == 10
        assert output_tokens == 20


class TestBaseCustomLLM:
    """Tests for BaseCustomLLM."""

    def test_setup_ssl_verify_enabled(self, mocker: MockerFixture) -> None:
        """Test SSL verification enabled by default."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {"SSL_CERTIFI_BUNDLE": "/path/to/bundle.pem"})

        BaseCustomLLM("gpt-4", {})

        assert mock_litellm.ssl_verify == "/path/to/bundle.pem"

    def test_setup_ssl_verify_disabled(self, mocker: MockerFixture) -> None:
        """Test SSL verification can be disabled."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})

        BaseCustomLLM("gpt-4", {"ssl_verify": False})

        assert mock_litellm.ssl_verify is False

    def test_setup_drop_params_enabled_by_default(self, mocker: MockerFixture) -> None:
        """Test drop_params is enabled by default."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})

        BaseCustomLLM("gpt-4", {})

        assert mock_litellm.drop_params is True

    def test_setup_drop_params_disabled(self, mocker: MockerFixture) -> None:
        """Test drop_params can be disabled."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})

        BaseCustomLLM("gpt-4", {"drop_params": False})

        assert mock_litellm.drop_params is False

    def test_setup_drop_params_enabled_explicitly(self, mocker: MockerFixture) -> None:
        """Test drop_params can be explicitly enabled."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})

        BaseCustomLLM("gpt-4", {"drop_params": True})

        assert mock_litellm.drop_params is True

    def test_call_returns_single_response(self, mocker: MockerFixture) -> None:
        """Test call returns single string when n=1."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})

        # Mock response
        mock_choice = mocker.Mock()
        mock_choice.message.content = "Test response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_litellm.completion.return_value = mock_response

        llm = BaseCustomLLM("gpt-4", {"temperature": 0.0})
        result = llm.call("test prompt")

        assert result == "Test response"

    def test_call_with_temperature_override(self, mocker: MockerFixture) -> None:
        """Test call with temperature override."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})

        mock_choice = mocker.Mock()
        mock_choice.message.content = "Test"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_litellm.completion.return_value = mock_response

        llm = BaseCustomLLM("gpt-4", {"temperature": 0.0})
        llm.call("test", temperature=0.9)

        call_args = mock_litellm.completion.call_args[1]
        assert call_args["temperature"] == 0.9

    def test_call_raises_llm_error_on_failure(self, mocker: MockerFixture) -> None:
        """Test call raises LLMError on failure."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})
        mock_litellm.completion.side_effect = Exception("API Error")

        llm = BaseCustomLLM("gpt-4", {})

        with pytest.raises(LLMError, match="LLM call failed"):
            llm.call("test prompt")
