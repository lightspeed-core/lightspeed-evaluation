"""Global litellm patching for token tracking.

It patches litellm.completion and litellm.acompletion to automatically track tokens
for all LLM calls throughout the application.
"""

import logging
from functools import wraps
from typing import Any

import litellm

from lightspeed_evaluation.core.llm.token_tracker import track_tokens

logger = logging.getLogger(__name__)


# Store original functions before patching
_original_completion = litellm.completion
_original_acompletion = litellm.acompletion


@wraps(_original_completion)
def _completion_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.completion that tracks tokens."""
    response = _original_completion(*args, **kwargs)
    try:
        track_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for completion: %s", e)
    return response


@wraps(_original_acompletion)
async def _acompletion_with_token_tracking(*args: Any, **kwargs: Any) -> Any:
    """Wrapper around litellm.acompletion that tracks tokens."""
    response = await _original_acompletion(*args, **kwargs)
    try:
        track_tokens(response)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to track tokens for acompletion: %s", e)
    return response


# Patch litellm's completion functions to include token tracking
litellm.completion = _completion_with_token_tracking
litellm.acompletion = _acompletion_with_token_tracking
