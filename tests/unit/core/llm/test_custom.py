# pylint: disable=protected-access,disable=too-few-public-methods

"""Unit tests for custom LLM classes."""

import threading

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.llm.custom import BaseCustomLLM, TokenTracker
from lightspeed_evaluation.core.system.exceptions import LLMError


class TestTokenTracker:
    """Tests for TokenTracker."""

    def test_add_tokens_accumulates(self) -> None:
        """Test that add_tokens accumulates token counts."""
        tracker = TokenTracker()

        tracker.add_tokens(10, 20)
        tracker.add_tokens(5, 15)

        input_tokens, output_tokens = tracker.get_counts()
        assert input_tokens == 15
        assert output_tokens == 35

    def test_reset_clears_counts(self) -> None:
        """Test that reset clears token counts."""
        tracker = TokenTracker()
        tracker.add_tokens(100, 200)

        tracker.reset()

        input_tokens, output_tokens = tracker.get_counts()
        assert input_tokens == 0
        assert output_tokens == 0

    def test_start_sets_active_tracker(self) -> None:
        """Test that start sets the tracker as active for current thread."""
        tracker = TokenTracker()
        tracker.start()

        try:
            assert TokenTracker.get_active() is tracker
        finally:
            tracker.stop()

    def test_stop_clears_active_tracker(self) -> None:
        """Test that stop clears the active tracker."""
        tracker = TokenTracker()
        tracker.start()
        tracker.stop()

        assert TokenTracker.get_active() is None

    def test_get_active_returns_none_when_no_tracker(self) -> None:
        """Test that get_active returns None when no tracker is active."""
        # Ensure clean state by starting and stopping a tracker
        temp = TokenTracker()
        temp.start()
        temp.stop()

        assert TokenTracker.get_active() is None

    def test_thread_local_isolation(self) -> None:
        """Test that each thread has its own active tracker."""
        tracker1 = TokenTracker()
        tracker2 = TokenTracker()
        results: dict[str, TokenTracker | None] = {}

        def thread_work(name: str, tracker: TokenTracker) -> None:
            tracker.start()
            results[name] = TokenTracker.get_active()
            # Deliberately don't stop to check isolation

        # Start tracker1 in main thread
        tracker1.start()

        # Start tracker2 in another thread
        thread = threading.Thread(target=thread_work, args=("thread2", tracker2))
        thread.start()
        thread.join()

        # Main thread should still have tracker1
        assert TokenTracker.get_active() is tracker1
        # Other thread had tracker2
        assert results["thread2"] is tracker2

        tracker1.stop()

    def test_add_tokens_thread_safe(self) -> None:
        """Test that add_tokens is thread-safe under concurrent access."""
        tracker = TokenTracker()
        num_threads = 10
        tokens_per_thread = 100

        def add_tokens_worker() -> None:
            for _ in range(tokens_per_thread):
                tracker.add_tokens(1, 2)

        threads = [
            threading.Thread(target=add_tokens_worker) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        input_tokens, output_tokens = tracker.get_counts()
        assert input_tokens == num_threads * tokens_per_thread
        assert output_tokens == num_threads * tokens_per_thread * 2


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

    def test_drop_params_always_enabled(self, mocker: MockerFixture) -> None:
        """Test drop_params is always enabled for cross-provider compatibility."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
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

    def test_call_captures_tokens_with_active_tracker(
        self, mocker: MockerFixture
    ) -> None:
        """Test call captures tokens when a TokenTracker is active."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
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
        mock_litellm.completion.return_value = mock_response

        # Start a tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            llm = BaseCustomLLM("gpt-4", {"temperature": 0.0})
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
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
        mocker.patch.dict("os.environ", {})

        # Mock response with usage
        mock_choice = mocker.Mock()
        mock_choice.message.content = "Test response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mocker.Mock()
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 100
        mock_litellm.completion.return_value = mock_response

        # Ensure no tracker is active
        temp = TokenTracker()
        temp.start()
        temp.stop()

        llm = BaseCustomLLM("gpt-4", {"temperature": 0.0})
        result = llm.call("test prompt")

        # Should succeed without error
        assert result == "Test response"

    def test_call_does_not_add_tokens_on_cache_hit(self, mocker: MockerFixture) -> None:
        """Test call does not add tokens when response is from cache."""
        mock_litellm = mocker.patch("lightspeed_evaluation.core.llm.custom.litellm")
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
        mock_litellm.completion.return_value = mock_response

        # Start a tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            llm = BaseCustomLLM("gpt-4", {"temperature": 0.0})
            llm.call("test prompt")

            # Tokens should NOT be captured due to cache hit
            input_tokens, output_tokens = tracker.get_counts()
            assert input_tokens == 0
            assert output_tokens == 0
        finally:
            tracker.stop()
