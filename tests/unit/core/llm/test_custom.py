# pylint: disable=protected-access,disable=too-few-public-methods

"""Unit tests for custom LLM classes."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.llm.custom import BaseCustomLLM
from lightspeed_evaluation.core.llm.token_tracker import TokenTracker
from lightspeed_evaluation.core.system.exceptions import LLMError


class TestBaseCustomLLM:
    """Tests for BaseCustomLLM."""

    def test_setup_ssl_verify_enabled(self, mocker: MockerFixture) -> None:
        """Test SSL verification enabled by default."""
        # Mock litellm in litellm_patch where setup_litellm_ssl is defined
        mock_litellm = mocker.patch(
            "lightspeed_evaluation.core.llm.litellm_patch.litellm"
        )
        mocker.patch.dict("os.environ", {"SSL_CERTIFI_BUNDLE": "/path/to/bundle.pem"})

        BaseCustomLLM("gpt-4", {})

        assert mock_litellm.ssl_verify == "/path/to/bundle.pem"

    def test_setup_ssl_verify_disabled(self, mocker: MockerFixture) -> None:
        """Test SSL verification can be disabled."""
        # Mock litellm in litellm_patch where setup_litellm_ssl is defined
        mock_litellm = mocker.patch(
            "lightspeed_evaluation.core.llm.litellm_patch.litellm"
        )
        mocker.patch.dict("os.environ", {})

        BaseCustomLLM("gpt-4", {"ssl_verify": False})

        assert mock_litellm.ssl_verify is False

    def test_drop_params_always_enabled(self, mocker: MockerFixture) -> None:
        """Test drop_params is always enabled for cross-provider compatibility."""
        # Mock litellm in custom.py where drop_params is set
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        # Also mock litellm_patch to prevent side effects
        mocker.patch("lightspeed_evaluation.core.llm.litellm_patch.litellm")
        mocker.patch.dict("os.environ", {})

        BaseCustomLLM("gpt-4", {})

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

        llm = BaseCustomLLM("gpt-4", {"parameters": {"temperature": 0.0}})
        result = llm.call("test prompt")

        assert result == "Test response"

    def test_call_raises_llm_error_on_failure(self, mocker: MockerFixture) -> None:
        """Test call raises LLMError on failure."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})
        mock_litellm.completion.side_effect = Exception("API Error")

        llm = BaseCustomLLM("gpt-4", {})

        with pytest.raises(LLMError, match="LLM call failed"):
            llm.call("test prompt")


class TestBaseCustomLLMJudgeLLMTokenTracking:
    """Tests for BaseCustomLLM JudgeLLM token tracking."""

    def test_call_captures_tokens_with_active_tracker(
        self, mocker: MockerFixture
    ) -> None:
        """Test call captures tokens when a TokenTracker is active."""
        mocker.patch.dict("os.environ", {})

        # Mock response with usage
        mock_choice = mocker.Mock()
        mock_choice.message.content = "Test response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mocker.Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response._hidden_params = {}  # Ensure no cache hit

        # Mock the ORIGINAL completion function, not the whole litellm module
        mocker.patch(
            "lightspeed_evaluation.core.llm.litellm_patch._original_completion",
            return_value=mock_response,
        )

        # Start a tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            llm = BaseCustomLLM("gpt-4", {"parameters": {"temperature": 0.0}})
            llm.call("test prompt")

            # Tokens should be captured
            input_tokens, output_tokens = tracker.get_counts()
            assert input_tokens == 50
            assert output_tokens == 100
        finally:
            tracker.stop()

    def test_call_does_not_capture_tokens_without_active_tracker(
        self, mocker: MockerFixture
    ) -> None:
        """Test call does not fail when no TokenTracker is active."""
        mocker.patch.dict("os.environ", {})

        # Mock response with usage
        mock_choice = mocker.Mock()
        mock_choice.message.content = "Test response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mocker.Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response._hidden_params = {}  # Ensure no cache hit

        # Mock the ORIGINAL completion function, not the whole litellm module
        mocker.patch(
            "lightspeed_evaluation.core.llm.litellm_patch._original_completion",
            return_value=mock_response,
        )

        # Ensure no tracker is active
        temp = TokenTracker()
        temp.start()
        temp.stop()

        llm = BaseCustomLLM("gpt-4", {"parameters": {"temperature": 0.0}})
        result = llm.call("test prompt")

        # Should succeed without error
        assert result == "Test response"

    def test_call_does_not_add_tokens_on_cache_hit(self, mocker: MockerFixture) -> None:
        """Test call does not add tokens when response is from cache."""
        mocker.patch.dict("os.environ", {})

        # Mock response with cache hit
        mock_choice = mocker.Mock()
        mock_choice.message.content = "Cached response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mocker.Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_response._hidden_params = {"cache_hit": True}  # Cache hit

        # Mock the ORIGINAL completion function to test cache-hit logic in track_tokens
        mocker.patch(
            "lightspeed_evaluation.core.llm.litellm_patch._original_completion",
            return_value=mock_response,
        )

        # Start a tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            llm = BaseCustomLLM("gpt-4", {"parameters": {"temperature": 0.0}})
            llm.call("test prompt")

            # Tokens should NOT be captured due to cache hit
            input_tokens, output_tokens = tracker.get_counts()
            assert input_tokens == 0
            assert output_tokens == 0
        finally:
            tracker.stop()
