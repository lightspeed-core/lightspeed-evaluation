"""LiteLLM configuration for token tracking and Ragas 0.4 compatibility.

This module configures litellm for two purposes:

1. TOKEN TRACKING: Wraps litellm.completion, litellm.acompletion, litellm.embedding,
   and litellm.aembedding to track token usage for all LLM and embedding calls.
   We use function wrapping rather than litellm's callback system because callbacks
   don't reliably capture tokens in all execution paths.

2. RAGAS 0.4 COMPATIBILITY: Ragas 0.4's score() method internally uses
   asyncio.run() which creates a new event loop. LiteLLM's background
   LoggingWorker task conflicts with this, causing:
   "RuntimeError: Queue is bound to a different event loop"

   We replace the LoggingWorker with a no-op implementation to avoid this.
   This is safe because we don't use litellm's built-in observability features.
"""

import logging
import os
import threading
import warnings
from functools import wraps
from typing import Any

import litellm

# Suppress coroutine warnings from litellm's async logging (cosmetic only)
warnings.filterwarnings(
    "ignore",
    message="coroutine.*was never awaited",
    category=RuntimeWarning,
)

# pylint: disable=wrong-import-position
from lightspeed_evaluation.core.llm.token_tracker import (  # noqa: E402
    track_judge_tokens,
    track_embedding_tokens,
)

logger = logging.getLogger(__name__)


# =============================================================================
# RAGAS 0.4 COMPATIBILITY: No-op logging worker
# =============================================================================
# Replace litellm's LoggingWorker with a no-op to prevent event loop conflicts
# when Ragas creates new event loops via asyncio.run().


class _NoOpLoggingWorker:
    """No-op logging worker to prevent event loop conflicts with Ragas 0.4.

    LiteLLM's LoggingWorker runs async tasks that conflict with Ragas's use of
    asyncio.run(). This no-op replacement silently ignores all logging operations.

    See: https://github.com/BerriAI/litellm/issues/17813
    """

    def ensure_initialized_and_enqueue(self, *args: Any, **kwargs: Any) -> None:
        """No-op: silently ignore."""

    def enqueue(self, *args: Any, **kwargs: Any) -> None:
        """No-op: silently ignore."""

    def start(self) -> None:
        """No-op: nothing to start."""

    def stop(self) -> None:
        """No-op: nothing to stop."""

    def flush(self) -> None:
        """No-op: nothing to flush."""

    def clear_queue(self) -> None:
        """No-op: nothing to clear."""


# Apply the no-op worker
try:
    # pylint: disable=ungrouped-imports
    import litellm.litellm_core_utils.logging_worker as logging_worker_module

    logging_worker_module.GLOBAL_LOGGING_WORKER = _NoOpLoggingWorker()  # type: ignore[assignment]
except (ImportError, AttributeError):
    pass  # Older versions of litellm may not have this

# Configure litellm to minimize async logging activity
litellm.suppress_debug_info = True


# =============================================================================
# TOKEN TRACKING: Wrap completion and embedding functions
# =============================================================================
# We wrap the completion and embedding functions rather than using callbacks
# because callbacks don't reliably capture tokens in all execution paths.

_original_completion = litellm.completion
_original_acompletion = litellm.acompletion
_original_embedding = litellm.embedding
_original_aembedding = litellm.aembedding


# Patch litellm's completion functions to include token tracking
@wraps(_original_completion)
def _completion_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.completion that tracks tokens."""
    response = _original_completion(*args, **kwargs)
    try:
        track_judge_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for completion: %s", e)
    return response


@wraps(_original_acompletion)
async def _acompletion_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.acompletion that tracks tokens."""
    response = await _original_acompletion(*args, **kwargs)
    try:
        track_judge_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for acompletion: %s", e)
    return response


# Patch litellm's embedding functions to include token tracking
@wraps(_original_embedding)
def _embedding_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.embedding that tracks tokens."""
    response = _original_embedding(*args, **kwargs)
    try:
        track_embedding_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for embedding: %s", e)
    return response


@wraps(_original_aembedding)
async def _aembedding_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.aembedding that tracks tokens."""
    response = await _original_aembedding(*args, **kwargs)
    try:
        track_embedding_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for aembedding: %s", e)
    return response


# Patch litellm's completion and embedding functions to include token tracking
litellm.completion = _completion_with_token_tracking
litellm.acompletion = _acompletion_with_token_tracking
litellm.embedding = _embedding_with_token_tracking
litellm.aembedding = _aembedding_with_token_tracking


# =============================================================================
# GLOBAL STATE LOCK
# =============================================================================
# Single lock for ALL litellm global state mutations (cache, ssl_verify).
# Import this lock in any module that reads/writes litellm.cache or
# litellm.ssl_verify to prevent race conditions between concurrent pipelines.
litellm_state_lock = threading.Lock()


# =============================================================================
# SSL CONFIGURATION UTILITY
# =============================================================================
def setup_litellm_ssl(llm_params: dict[str, Any]) -> None:
    """Configure litellm SSL verification.

    Args:
        llm_params: Dictionary containing LLM parameters including 'ssl_verify'
    """
    ssl_verify = llm_params.get("ssl_verify", True)

    with litellm_state_lock:
        if ssl_verify:
            litellm.ssl_verify = os.environ.get("SSL_CERTIFI_BUNDLE", True)
        else:
            litellm.ssl_verify = False
