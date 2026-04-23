"""Unit tests for TokenTracker and integration with litellm patch."""

import threading
from typing import Any, Callable

import pytest
from pytest_mock import MockerFixture
import litellm

# Simulate litellm completion call through patch
from lightspeed_evaluation.core.llm import litellm_patch
from lightspeed_evaluation.core.llm.token_tracker import (
    TokenTracker,
    _extract_tokens_if_not_cached,
)


class TestTokenTracker:
    """Tests for TokenTracker."""

    def test_add_tokens_accumulates(self) -> None:
        """Test that add tokens functions accumulate token counts."""
        tracker = TokenTracker()

        tracker.add_judge_tokens(10, 20)
        tracker.add_judge_tokens(5, 15)
        tracker.add_embedding_tokens(7)

        judge_input_tokens, judge_output_tokens = tracker.get_judge_counts()
        embed_input_tokens = tracker.get_embedding_counts()
        assert judge_input_tokens == 15
        assert judge_output_tokens == 35
        assert embed_input_tokens == 7

    def test_reset_clears_counts(self) -> None:
        """Test that reset clears token counts."""
        tracker = TokenTracker()
        tracker.add_judge_tokens(100, 200)
        tracker.add_embedding_tokens(75)

        tracker.reset()

        judge_input_tokens, judge_output_tokens = tracker.get_judge_counts()
        embed_input_tokens = tracker.get_embedding_counts()
        assert judge_input_tokens == 0
        assert judge_output_tokens == 0
        assert embed_input_tokens == 0

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

    def test_double_start_is_safe(self) -> None:
        """Test that calling start twice doesn't fail or cause issues."""
        tracker = TokenTracker()
        tracker.start()
        tracker.start()  # Should not fail

        try:
            assert TokenTracker.get_active() is tracker
        finally:
            tracker.stop()

    def test_double_stop_is_safe(self) -> None:
        """Test that calling stop twice doesn't fail."""
        tracker = TokenTracker()
        tracker.start()
        tracker.stop()
        tracker.stop()  # Should not fail

        assert TokenTracker.get_active() is None

    def test_independent_tracker_instances(self) -> None:
        """Test that multiple TokenTracker instances maintain independent counts."""
        tracker1 = TokenTracker()
        tracker2 = TokenTracker()

        tracker1.add_judge_tokens(100, 50)
        tracker2.add_judge_tokens(200, 100)

        assert tracker1.get_judge_counts() == (100, 50)
        assert tracker2.get_judge_counts() == (200, 100)

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

    def test_add_judge_tokens_thread_safe(self) -> None:
        """Test that add tokens is thread-safe under concurrent access."""
        tracker = TokenTracker()
        num_threads = 10
        tokens_per_thread = 100

        def add_tokens_worker() -> None:
            for _ in range(tokens_per_thread):
                tracker.add_judge_tokens(1, 2)
                tracker.add_embedding_tokens(1)

        threads = [
            threading.Thread(target=add_tokens_worker) for _ in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        judge_input_tokens, judge_output_tokens = tracker.get_judge_counts()
        embed_input_tokens = tracker.get_embedding_counts()
        assert judge_input_tokens == num_threads * tokens_per_thread
        assert judge_output_tokens == num_threads * tokens_per_thread * 2
        assert embed_input_tokens == num_threads * tokens_per_thread


class TestTokenTrackerCacheHandling:
    """Test token extraction with cache handling."""

    def test_extract_tokens_if_not_cached_cache_miss(
        self, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that tokens are extracted when response is not from cache."""
        mocked_llm_response = mock_judge_llm_response(
            prompt_tokens=200,
            completion_tokens=100,
            cache_hit=False,
            content="Test response",
        )

        tokens = _extract_tokens_if_not_cached(mocked_llm_response)
        assert tokens
        prompt_tokens, completion_tokens = tokens

        # Token counts are retrieved in case of a cache miss
        assert prompt_tokens == 200
        assert completion_tokens == 100

    def test_extract_tokens_if_not_cached_cache_hit(
        self, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that None is returned when response is from cache."""
        mocked_llm_response = mock_judge_llm_response(
            prompt_tokens=200,
            completion_tokens=100,
            cache_hit=True,
            content="Test response",
        )

        tokens = _extract_tokens_if_not_cached(mocked_llm_response)

        # Tokens are not counted for cache hit
        assert not tokens


class TestLiteLLMCompletionPatchWithTokenTracker:
    """Test LiteLLM completion/acompletion patch integration with TokenTracker."""

    def test_tracks_tokens_via_llm_completion_patch(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that tokens are tracked via litellm completion patch when tracker is active."""
        # Mock the original completion function
        mock_completion = mocker.patch(f"{litellm_patch.__name__}._original_completion")
        mock_completion.return_value = mock_judge_llm_response(
            prompt_tokens=100,
            completion_tokens=50,
            cache_hit=False,
            content="Test response",
        )

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            litellm.completion(
                model="gpt-4", messages=[{"role": "user", "content": "test"}]
            )

            # Verify tokens were tracked
            input_tokens, output_tokens = tracker.get_judge_counts()
            assert input_tokens == 100
            assert output_tokens == 50
        finally:
            tracker.stop()

    @pytest.mark.asyncio
    async def test_tracks_tokens_via_llm_acompletion_patch(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that tokens are tracked via litellm acompletion patch when tracker is active."""
        # Mock the original acompletion function
        mock_acompletion = mocker.patch(
            f"{litellm_patch.__name__}._original_acompletion"
        )
        mock_acompletion.return_value = mock_judge_llm_response(
            prompt_tokens=100,
            completion_tokens=50,
            cache_hit=False,
            content="Test response",
        )

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            await litellm.acompletion(
                model="gpt-4", messages=[{"role": "user", "content": "test"}]
            )

            # Verify tokens were tracked
            input_tokens, output_tokens = tracker.get_judge_counts()
            assert input_tokens == 100
            assert output_tokens == 50
        finally:
            tracker.stop()

    def test_track_judge_tokens_exception_does_not_break_response(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that exceptions in token tracking do not prevent response from being returned."""
        mock_completion = mocker.patch(f"{litellm_patch.__name__}._original_completion")
        mock_completion.return_value = mock_judge_llm_response(
            prompt_tokens="not_a_number",
            completion_tokens="not_a_number",
            cache_hit=False,
            content="Test response",
        )

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            # Should not raise despite track_judge_tokens failing internally
            response = litellm.completion(
                model="gpt-4", messages=[{"role": "user", "content": "test"}]
            )

            # Verify response was returned successfully
            assert response.choices[0].message.content == "Test response"

            # Tokens should not be tracked due to exception
            input_tokens, output_tokens = tracker.get_judge_counts()
            assert input_tokens == 0
            assert output_tokens == 0
        finally:
            tracker.stop()


class TestLiteLLMEmbeddingPatchWithTokenTracker:
    """Test LiteLLM embedding/aembedding patches integration with TokenTracker."""

    def test_tracks_tokens_via_embedding_patch(
        self, mocker: MockerFixture, mock_embedding_response: Callable[..., Any]
    ) -> None:
        """Test that tokens are tracked via litellm embedding patch when tracker is active."""
        # Mock the original embedding function
        mock_embedding = mocker.patch(f"{litellm_patch.__name__}._original_embedding")
        mock_embedding.return_value = mock_embedding_response(
            prompt_tokens=10, cache_hit=False, embedding=[0.1, 0.2, 0.3]
        )

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            litellm.embedding(model="text-embedding-3-small", input="test")

            # Verify tokens were tracked
            input_tokens = tracker.get_embedding_counts()
            assert input_tokens == 10
        finally:
            tracker.stop()

    @pytest.mark.asyncio
    async def test_tracks_tokens_via_aembedding_patch(
        self, mocker: MockerFixture, mock_embedding_response: Callable[..., Any]
    ) -> None:
        """Test that tokens are tracked via litellm aembedding patch when tracker is active."""
        # Mock the original aembedding function
        mock_embedding = mocker.patch(f"{litellm_patch.__name__}._original_aembedding")
        mock_embedding.return_value = mock_embedding_response(
            prompt_tokens=10, cache_hit=False, embedding=[0.1, 0.2, 0.3]
        )

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            await litellm.aembedding(
                model="text-embedding-3-small",
                input="async test",
            )

            # Verify tokens were tracked
            input_tokens = tracker.get_embedding_counts()
            assert input_tokens == 10
        finally:
            tracker.stop()

    def test_track_embedding_tokens_exception_does_not_break_response(
        self, mocker: MockerFixture, mock_embedding_response: Callable[..., Any]
    ) -> None:
        """Test that exceptions in track_embedding_tokens do not prevent response."""
        # Mock the original embedding function
        mock_embedding = mocker.patch(f"{litellm_patch.__name__}._original_embedding")
        mock_embedding.return_value = mock_embedding_response(
            prompt_tokens="not_a_number", cache_hit=False, embedding=[0.1, 0.2, 0.3]
        )

        # Start tracker
        tracker = TokenTracker()
        tracker.start()

        try:
            response = litellm.embedding(model="text-embedding-3-small", input="test")

            # Verify response was returned successfully
            assert response.object == "list"
            assert len(response.data) == 1
            assert response.data[0].embedding == [0.1, 0.2, 0.3]

            # Tokens should not be tracked due to exception
            input_tokens = tracker.get_embedding_counts()
            assert input_tokens == 0
        finally:
            tracker.stop()
