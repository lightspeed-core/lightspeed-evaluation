"""TokenTracker for tracking LLM token usage with direct response extraction."""

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Thread-local storage for active TokenTracker
_active_tracker: threading.local = threading.local()


class TokenTracker:
    """Tracks token usage from LLM calls using direct response extraction.

    Uses thread-local storage to track the active tracker. Tokens are captured
    directly from litellm response in BaseCustomLLM.call() - no callbacks,
    no timeouts, no race conditions.

    Usage:
        tracker = TokenTracker()
        tracker.start()  # Set as active tracker for this thread
        # ... make LLM calls (tokens captured automatically) ...
        tracker.stop()   # Unset as active tracker
        input_tokens, output_tokens = tracker.get_judge_counts()
    """

    def __init__(self) -> None:
        """Initialize token tracker."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.embedding_tokens = 0
        self._lock = threading.Lock()  # Instance lock for token counter updates

    def add_judge_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Add judge token counts (thread-safe).

        Called by the litellm patch (see litellm_patch.py) after each completions call.

        Args:
            prompt_tokens: Number of input/prompt tokens.
            completion_tokens: Number of output/completion tokens.
        """
        with self._lock:
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens

    def add_embedding_tokens(self, prompt_tokens: int) -> None:
        """Add embedding token counts (thread-safe).

        Called by the litellm patch (see litellm_patch.py) after each embedding call.

        Args:
            prompt_tokens: Number of input/prompt tokens for embeddings.
        """
        with self._lock:
            self.embedding_tokens += prompt_tokens

    def start(self) -> None:
        """Set this tracker as active for the current thread."""
        _active_tracker.tracker = self

    def stop(self) -> None:
        """Unset this tracker as active for the current thread."""
        if getattr(_active_tracker, "tracker", None) is self:
            _active_tracker.tracker = None

    def get_judge_counts(self) -> tuple[int, int]:
        """Get accumulated token counts.

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        with self._lock:
            return self.input_tokens, self.output_tokens

    def get_embedding_counts(self) -> int:
        """Get accumulated embedding token counts.

        Returns:
            embedding_tokens (embeddings don't have output tokens)
        """
        with self._lock:
            return self.embedding_tokens

    def reset(self) -> None:
        """Reset token counts to zero."""
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0
            self.embedding_tokens = 0

    @staticmethod
    def get_active() -> Optional["TokenTracker"]:
        """Get the active tracker for the current thread.

        Returns:
            The active TokenTracker, or None if no tracker is active.
        """
        return getattr(_active_tracker, "tracker", None)


def _extract_tokens_if_not_cached(response: Any) -> tuple[int, int] | None:
    """Extract tokens from response if not cached. Returns (prompt, completion) or None."""
    cache_hit = getattr(
        response, "_hidden_params", {}
    ).get(  # pylint: disable=protected-access
        "cache_hit", False
    )

    # Only add tokens if this response was not retrieved from cache
    if not cache_hit and hasattr(response, "usage") and response.usage:
        prompt_tokens = int(getattr(response.usage, "prompt_tokens", 0))
        completion_tokens = int(getattr(response.usage, "completion_tokens", 0))
        return (prompt_tokens, completion_tokens)
    return None


def track_judge_tokens(response: Any) -> None:
    """Track JudgeLLM tokens if a tracker is active."""
    tracker = TokenTracker.get_active()
    if tracker and response is not None:
        tokens = _extract_tokens_if_not_cached(response)
        if tokens:
            prompt_tokens, completion_tokens = tokens
            tracker.add_judge_tokens(prompt_tokens, completion_tokens)


def track_embedding_tokens(response: Any) -> None:
    """Track embedding tokens if a tracker is active."""
    tracker = TokenTracker.get_active()
    if tracker and response is not None:
        tokens = _extract_tokens_if_not_cached(response)
        if tokens:
            prompt_tokens, _ = tokens
            tracker.add_embedding_tokens(prompt_tokens)
