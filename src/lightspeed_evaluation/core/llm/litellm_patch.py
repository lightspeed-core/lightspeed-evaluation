"""LiteLLM configuration for token tracking, Ragas 0.4 compatibility, and Vertex AI support.

This module configures litellm for three purposes:

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

3. VERTEX AI PER-MODEL REGION SUPPORT: litellm.drop_params=True (set by
   DeepEval) silently strips vertex_project and vertex_location from
   completion kwargs. The completion wrappers intercept these params and
   temporarily set them as litellm module-level attributes, which litellm
   checks as a fallback in its vertex_ai handler.
"""

import asyncio
import logging
import os
import threading
import warnings
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Generator

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
# GLOBAL STATE LOCK
# =============================================================================
# Single lock for ALL litellm global state mutations (cache, ssl_verify,
# vertex_project, vertex_location). Import this lock in any module that
# reads/writes litellm global state to prevent race conditions between
# concurrent pipelines. Both sync and async code paths share this lock;
# async callers use asyncio.to_thread so the event loop is never blocked.
litellm_state_lock = threading.Lock()


# =============================================================================
# VERTEX AI PER-MODEL REGION SUPPORT
# =============================================================================
# litellm.drop_params=True (set by DeepEval) silently strips vertex_project
# and vertex_location from completion kwargs. We intercept these params and
# temporarily set them as litellm module-level attributes, which litellm
# checks as a fallback in its vertex_ai handler.


@contextmanager
def _vertex_override(kwargs: dict[str, Any]) -> Generator[None, None, None]:
    """Pop vertex_project/vertex_location from kwargs and set as litellm module attrs.

    Always acquires litellm_state_lock to prevent concurrent reads of partially
    updated globals, even when no vertex params are present in kwargs.
    """
    with litellm_state_lock:
        vp = kwargs.pop("vertex_project", None)
        vl = kwargs.pop("vertex_location", None)
        if vp is None and vl is None:
            yield
            return
        old_vp = getattr(litellm, "vertex_project", None)
        old_vl = getattr(litellm, "vertex_location", None)
        try:
            if vp is not None:
                litellm.vertex_project = vp
            if vl is not None:
                litellm.vertex_location = vl
            yield
        finally:
            litellm.vertex_project = old_vp
            litellm.vertex_location = old_vl


@asynccontextmanager
async def _vertex_override_async(
    kwargs: dict[str, Any],
) -> AsyncGenerator[None, None]:
    """Async version of _vertex_override using asyncio.to_thread.

    Runs lock acquisition and litellm global-state mutation in a thread-pool
    worker via asyncio.to_thread so the event loop is never blocked.  Uses the
    same litellm_state_lock as the synchronous path to prevent races between
    sync and async callers.
    """

    def _apply() -> tuple[Any, Any] | None:
        with litellm_state_lock:
            vp = kwargs.pop("vertex_project", None)
            vl = kwargs.pop("vertex_location", None)
            if vp is None and vl is None:
                return None
            old_vp = getattr(litellm, "vertex_project", None)
            old_vl = getattr(litellm, "vertex_location", None)
            if vp is not None:
                litellm.vertex_project = vp
            if vl is not None:
                litellm.vertex_location = vl
            return (old_vp, old_vl)

    def _restore(old: tuple[Any, Any]) -> None:
        with litellm_state_lock:
            litellm.vertex_project = old[0]
            litellm.vertex_location = old[1]

    old = await asyncio.to_thread(_apply)
    if old is None:
        yield
        return
    try:
        yield
    finally:
        await asyncio.to_thread(_restore, old)


# =============================================================================
# TOKEN TRACKING: Wrap completion and embedding functions
# =============================================================================
# We wrap the completion and embedding functions rather than using callbacks
# because callbacks don't reliably capture tokens in all execution paths.

_original_completion = litellm.completion
_original_acompletion = litellm.acompletion
_original_embedding = litellm.embedding
_original_aembedding = litellm.aembedding


@wraps(_original_completion)
def _completion_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.completion that tracks tokens and handles Vertex params."""
    with _vertex_override(kwargs):
        response = _original_completion(*args, **kwargs)
    try:
        track_judge_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for completion: %s", e)
    return response


@wraps(_original_acompletion)
async def _acompletion_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.acompletion that tracks tokens and handles Vertex params."""
    async with _vertex_override_async(kwargs):
        response = await _original_acompletion(*args, **kwargs)
    try:
        track_judge_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for acompletion: %s", e)
    return response


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


litellm.completion = _completion_with_token_tracking
litellm.acompletion = _acompletion_with_token_tracking
litellm.embedding = _embedding_with_token_tracking
litellm.aembedding = _aembedding_with_token_tracking


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
