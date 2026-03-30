"""Ragas LLM Manager - Ragas-specific LLM wrapper using llm_factory."""

import logging
from typing import Any

import litellm
from ragas.llms import llm_factory

from lightspeed_evaluation.core.llm.litellm_patch import setup_litellm_ssl
from lightspeed_evaluation.core.llm.manager import LLMManager

logger = logging.getLogger(__name__)


class RagasLLMManager:
    """Ragas LLM Manager - Creates LLM for Ragas 0.4+ metrics using llm_factory.

    This manager uses ragas's llm_factory with litellm provider to create an
    InstructorLLM that is compatible with ragas 0.4+ collections-based metrics.

    IMPORTANT: Ragas 0.4+ metrics internally use async operations even when
    calling sync methods like score(). We must use litellm.acompletion (async)
    instead of litellm.completion (sync) to avoid the error:
    "Cannot use agenerate() with a synchronous client"

    NOTE: Async logging is disabled in litellm_patch.py to prevent event loop
    conflicts when ragas creates new event loops via asyncio.run().
    """

    def __init__(self, llm_manager: LLMManager):
        """Initialize with LLM parameters from LLMManager.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters
        """
        self.model_name = llm_manager.get_model_name()
        self.llm_params = llm_manager.get_llm_params()

        # Always drop unsupported parameters for cross-provider compatibility
        litellm.drop_params = True

        # Setup SSL verification for litellm
        setup_litellm_ssl(self.llm_params)

        # Build inference kwargs from parameters
        # Rename max_completion_tokens to max_tokens for ragas/instructor compatibility
        # (OpenAI rejects requests with both set simultaneously)
        # Note: Forbidden keys are rejected at LLMParametersConfig load time
        inference_kwargs = dict(self.llm_params.get("parameters", {}))
        if "max_completion_tokens" in inference_kwargs:
            inference_kwargs["max_tokens"] = inference_kwargs.pop(
                "max_completion_tokens"
            )

        # Create LLM using ragas llm_factory with litellm provider
        # MUST use acompletion (async) because ragas 0.4 metrics use async internally
        self.llm = llm_factory(
            provider="litellm",
            client=litellm.acompletion,
            model=self.model_name,
            timeout=self.llm_params.get("timeout"),
            num_retries=self.llm_params.get("num_retries"),
            **inference_kwargs,
        )

        logger.info("Ragas LLM Manager configured with model: %s", self.model_name)

    def get_llm(self) -> Any:
        """Get the configured Ragas LLM (InstructorLLM).

        Returns:
            The InstructorLLM instance created by llm_factory
        """
        return self.llm

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        params = self.llm_params.get("parameters", {})
        info: dict[str, Any] = {"model_name": self.model_name}
        # Only include temperature if explicitly set (not removed via null)
        if "temperature" in params:
            info["temperature"] = params["temperature"]
        return info
