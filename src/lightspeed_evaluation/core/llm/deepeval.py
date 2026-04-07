"""DeepEval LLM Manager - DeepEval-specific LLM wrapper.

Note: litellm patching is applied at package level (__init__.py) before any imports.
This ensures DeepEval's LiteLLMModel uses the patched completion functions.
"""

import asyncio
import logging
from typing import Any

import litellm
from deepeval.models import LiteLLMModel

from lightspeed_evaluation.core.llm.litellm_patch import setup_litellm_ssl

logger = logging.getLogger(__name__)


class DeepEvalLLMManager:
    """DeepEval LLM Manager - Takes LLM parameters directly.

    This manager focuses solely on DeepEval-specific LLM integration
    with token tracking support.
    """

    def __init__(self, model_name: str, llm_params: dict[str, Any]):
        """Initialize with LLM parameters from LLMManager.

        Args:
            model_name: Constructed model name (e.g., "openai/gpt-4")
            llm_params: LLM params from get_llm_params() — operational fields
                plus a "parameters" dict with inference params.
        """
        self.model_name = model_name
        self.llm_params = llm_params

        self.setup_ssl_verify()

        # Always drop unsupported parameters for cross-provider compatibility
        litellm.drop_params = True

        # Note: Token tracking is handled by the patched litellm.completion/acompletion
        # No additional setup needed - the patch was applied at module import time
        # Create standard LiteLLMModel - it will use our patched completion functions

        # LiteLLMModel stores **kwargs in self.kwargs and merges them into
        # every litellm.completion() call
        # Note: Forbidden keys are rejected at LLMParametersConfig load time
        self.llm_model = LiteLLMModel(
            model=self.model_name,
            timeout=self.llm_params.get("timeout"),
            num_retries=self.llm_params.get("num_retries"),
            **self.llm_params.get("parameters", {}),
        )

        print(f"✅ DeepEval LLM Manager: {self.model_name}")

    def setup_ssl_verify(self) -> None:
        """Setup SSL verification based on LLM parameters.

        Delegates to the shared ``setup_litellm_ssl`` in ``litellm_patch``
        which acquires ``litellm_state_lock`` to serialize access.
        """
        setup_litellm_ssl(self.llm_params)

    def get_llm(self) -> LiteLLMModel:
        """Get the configured DeepEval LLM model."""
        return self.llm_model

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        info: dict[str, Any] = {"model_name": self.model_name}
        info["timeout"] = self.llm_params.get("timeout")
        info["num_retries"] = self.llm_params.get("num_retries")
        # Add inference params (forbidden keys rejected at config load time)
        info.update(self.llm_params.get("parameters", {}))
        return info

    @staticmethod
    def flush_deepevals_pending_tasks() -> None:
        """Flush background tasks left pending by DeepEvals async_mode."""
        try:
            asyncio.run(asyncio.sleep(0))
        except RuntimeError as e:
            logger.debug("Could not flush DeepEval pending tasks: %s", e)
