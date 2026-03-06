"""Unit tests for TokenTracker and integration with litellm patch."""

import threading

from pytest_mock import MockerFixture
import litellm

# Simulate litellm completion call through patch
from lightspeed_evaluation.core.llm import litellm_patch
from lightspeed_evaluation.core.llm.token_tracker import TokenTracker


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
            try:
                tracker.start()
                results[name] = TokenTracker.get_active()
                # Check isolation before stopping
            finally:
                tracker.stop()

        # Start tracker1 in main thread
        tracker1.start()

        try:
            # Start tracker2 in another thread
            thread = threading.Thread(target=thread_work, args=("thread2", tracker2))
            thread.start()
            thread.join()

            # Main thread should still have tracker1
            assert TokenTracker.get_active() is tracker1
            # Other thread had tracker2
            assert results["thread2"] is tracker2
        finally:
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


class TestLiteLLMPatchWithTokenTracker:
    """Test LiteLLM patch integration with TokenTracker."""

    def test_tracks_tokens_via_llm_patch(self, mocker: MockerFixture) -> None:
        """Test that tokens are tracked via litellm patch when tracker is active."""
        # Mock the original completion function
        mock_completion = mocker.patch(f"{litellm_patch.__name__}._original_completion")

        # Mock LLM response with usage
        mock_choice = mocker.Mock()
        mock_choice.message.content = "Test response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mocker.Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        setattr(mock_response, "_hidden_params", {})
        mock_completion.return_value = mock_response

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            litellm.completion(
                model="gpt-4", messages=[{"role": "user", "content": "test"}]
            )

            # Verify tokens were tracked
            input_tokens, output_tokens = tracker.get_counts()
            assert input_tokens == 100
            assert output_tokens == 50
        finally:
            tracker.stop()

    def test_skips_tokens_on_cache_hit(self, mocker: MockerFixture) -> None:
        """Test that cached responses do not add to token counts."""
        mock_completion = mocker.patch(f"{litellm_patch.__name__}._original_completion")

        # Mock cached LLM response
        mock_choice = mocker.Mock()
        mock_choice.message.content = "Cached response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mocker.Mock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100
        setattr(mock_response, "_hidden_params", {"cache_hit": True})  # Cache hit
        mock_completion.return_value = mock_response

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            litellm.completion(
                model="gpt-4", messages=[{"role": "user", "content": "test"}]
            )

            # Tokens should NOT be counted for cache hits
            input_tokens, output_tokens = tracker.get_counts()
            assert input_tokens == 0
            assert output_tokens == 0
        finally:
            tracker.stop()

    def test_track_tokens_exception_does_not_break_response(
        self, mocker: MockerFixture
    ) -> None:
        """Test that exceptions in track_tokens do not prevent response from being returned."""
        mock_completion = mocker.patch(f"{litellm_patch.__name__}._original_completion")

        mock_choice = mocker.Mock()
        mock_choice.message.content = "Test response"
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mocker.Mock()
        # This will cause int() to raise ValueError in track_tokens
        mock_response.usage.prompt_tokens = "not_a_number"
        mock_response.usage.completion_tokens = "not_a_number"
        setattr(mock_response, "_hidden_params", {})
        mock_completion.return_value = mock_response

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            # Should not raise despite track_tokens failing internally
            response = litellm.completion(
                model="gpt-4", messages=[{"role": "user", "content": "test"}]
            )

            # Verify response was returned successfully
            assert response is mock_response
            assert response.choices[0].message.content == "Test response"

            # Tokens should not be tracked due to exception
            input_tokens, output_tokens = tracker.get_counts()
            assert input_tokens == 0
            assert output_tokens == 0
        finally:
            tracker.stop()
