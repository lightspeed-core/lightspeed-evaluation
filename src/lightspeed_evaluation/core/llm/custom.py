"""Base Custom LLM class for evaluation framework."""

import os
import logging
import threading
from typing import Any, Optional, Union

import litellm
from litellm.exceptions import InternalServerError

from lightspeed_evaluation.core.system.exceptions import LLMError

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
        input_tokens, output_tokens = tracker.get_counts()
    """

    def __init__(self) -> None:
        """Initialize token tracker."""
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()  # Instance lock for token counter updates

    def add_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Add token counts (thread-safe).

        Called by BaseCustomLLM.call() to record tokens from LLM response.

        Args:
            prompt_tokens: Number of input/prompt tokens.
            completion_tokens: Number of output/completion tokens.
        """
        with self._lock:
            self.input_tokens += prompt_tokens
            self.output_tokens += completion_tokens

    def start(self) -> None:
        """Set this tracker as active for the current thread."""
        _active_tracker.tracker = self

    def stop(self) -> None:
        """Unset this tracker as active for the current thread."""
        if getattr(_active_tracker, "tracker", None) is self:
            _active_tracker.tracker = None

    def get_counts(self) -> tuple[int, int]:
        """Get accumulated token counts.

        Returns:
            Tuple of (input_tokens, output_tokens)
        """
        with self._lock:
            return self.input_tokens, self.output_tokens

    def reset(self) -> None:
        """Reset token counts to zero."""
        with self._lock:
            self.input_tokens = 0
            self.output_tokens = 0

    @staticmethod
    def get_active() -> Optional["TokenTracker"]:
        """Get the active tracker for the current thread.

        Returns:
            The active TokenTracker, or None if no tracker is active.
        """
        return getattr(_active_tracker, "tracker", None)


class BaseCustomLLM:  # pylint: disable=too-few-public-methods
    """Base LLM class with core calling functionality."""

    def __init__(self, model_name: str, llm_params: dict[str, Any]):
        """Initialize with model configuration."""
        self.model_name = model_name
        self.llm_params = llm_params

        self.setup_ssl_verify()

        # Always drop unsupported parameters for cross-provider compatibility
        litellm.drop_params = True

    def setup_ssl_verify(self) -> None:
        """Setup SSL verification based on LLM parameters."""
        ssl_verify = self.llm_params.get("ssl_verify", True)

        if ssl_verify:
            # Use our combined certifi bundle (includes system + custom certs)
            litellm.ssl_verify = os.environ.get("SSL_CERTIFI_BUNDLE", True)
        else:
            # Explicitly disable SSL verification
            litellm.ssl_verify = False

    def call(
        self,
        prompt: str,
        n: int = 1,
        temperature: Optional[float] = None,
        return_single: bool = True,
        **kwargs: Any,
    ) -> Union[str, list[str]]:
        """Make LLM call and return response(s).

        Args:
            prompt: Text prompt to send
            n: Number of responses to generate (default 1)
            temperature: Override temperature (uses config default if None)
            return_single: If True and n=1, return single string. If False, always return list.
            **kwargs: Additional LLM parameters

        Returns:
            Single string if return_single=True and n=1, otherwise list of strings
        """
        temp = (
            temperature
            if temperature is not None
            else self.llm_params.get("temperature", 0.0)
        )

        call_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "n": n,
            "max_completion_tokens": self.llm_params.get("max_completion_tokens"),
            "timeout": self.llm_params.get("timeout"),
            "num_retries": self.llm_params.get("num_retries", 3),
            **kwargs,
        }

        response = None
        try:
            response = litellm.completion(**call_params)

            # Extract content from all choices
            results = []
            for choice in response.choices:  # type: ignore
                content = choice.message.content  # type: ignore
                if content is None:
                    content = ""
                results.append(content.strip())

            # Return format based on parameters
            if return_single and n == 1:
                if not results:
                    raise LLMError("LLM returned empty response")
                return results[0]

            return results

        except InternalServerError as e:
            # Check if it's an SSL/certificate error
            error_msg = str(e)
            if "[X509]" in error_msg or "PEM lib" in error_msg:
                raise LLMError(
                    f"Judge LLM SSL certificate verification failed: {error_msg}"
                ) from e

            # Otherwise, it's a different internal server error
            raise LLMError(f"LLM internal server error: {error_msg}") from e

        except Exception as e:
            raise LLMError(f"LLM call failed: {str(e)}") from e

        finally:
            # Track tokens even if the call failed - tokens may have been consumed
            self._track_tokens(response)

    def _track_tokens(self, response: Any) -> None:
        """Track JudgeLLM tokens if a tracker is active."""
        # Only track token counts if response exists and is NOT from cache
        tracker = TokenTracker.get_active()
        if tracker and response is not None:
            cache_hit = getattr(
                response, "_hidden_params", {}
            ).get(  # pylint: disable=protected-access
                "cache_hit", False
            )
            # Only add tokens if this response was not retrieved from cache
            if not cache_hit and hasattr(response, "usage") and response.usage:
                tracker.add_tokens(
                    getattr(response.usage, "prompt_tokens", 0),
                    getattr(response.usage, "completion_tokens", 0),
                )
