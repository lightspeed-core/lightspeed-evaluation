"""DeepEval LLM Manager - DeepEval-specific LLM wrapper.

Note: litellm patching is applied at package level (__init__.py) before any imports.
This ensures DeepEval's LiteLLMModel uses the patched completion functions.
"""

import os
from typing import Any

import litellm
from deepeval.models import LiteLLMModel


class DeepEvalLLMManager:
    """DeepEval LLM Manager - Takes LLM parameters directly.

    This manager focuses solely on DeepEval-specific LLM integration
    with token tracking support.
    """

    def __init__(self, model_name: str, llm_params: dict[str, Any]):
        """Initialize with LLM parameters from LLMManager."""
        self.model_name = model_name
        self.llm_params = llm_params

        self.setup_ssl_verify()

        # Always drop unsupported parameters for cross-provider compatibility
        litellm.drop_params = True

        # Note: Token tracking is handled by the patched litellm.completion/acompletion
        # No additional setup needed - the patch was applied at module import time

        # Create standard LiteLLMModel - it will use our patched completion functions
        self.llm_model = LiteLLMModel(
            model=self.model_name,
            temperature=llm_params.get("temperature", 0.0),
            max_completion_tokens=llm_params.get("max_completion_tokens"),
            timeout=llm_params.get("timeout"),
            num_retries=llm_params.get("num_retries", 3),
        )

        print(f"✅ DeepEval LLM Manager: {self.model_name}")

    def setup_ssl_verify(self) -> None:
        """Setup SSL verification based on LLM parameters."""
        ssl_verify = self.llm_params.get(
            "ssl_verify", True
        )  # pylint: disable=duplicate-code

        if ssl_verify:
            # Use our combined certifi bundle (includes system + custom certs)
            litellm.ssl_verify = os.environ.get("SSL_CERTIFI_BUNDLE", True)
        else:
            # Explicitly disable SSL verification
            litellm.ssl_verify = False

    def get_llm(self) -> LiteLLMModel:
        """Get the configured DeepEval LLM model."""
        return self.llm_model

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        return {
            "model_name": self.model_name,
            "temperature": self.llm_params.get("temperature", 0.0),
            "max_completion_tokens": self.llm_params.get("max_completion_tokens"),
            "timeout": self.llm_params.get("timeout"),
            "num_retries": self.llm_params.get("num_retries", 3),
        }
