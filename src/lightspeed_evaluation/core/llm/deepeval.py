"""DeepEval LLM Manager - DeepEval-specific LLM wrapper that takes LiteLLM parameters."""

from typing import Any, Dict

from deepeval.models import LiteLLMModel


class DeepEvalLLMManager:
    """
    DeepEval LLM Manager - Takes LLM parameters directly.

    This manager focuses solely on DeepEval-specific LLM integration.
    """

    def __init__(self, model_name: str, litellm_params: Dict[str, Any]):
        """Initialize with LLM parameters from LLMManager."""
        self.model_name = model_name
        self.litellm_params = litellm_params

        # Create DeepEval's LiteLLMModel with provided parameters
        self.llm_model = LiteLLMModel(
            model=self.model_name,
            temperature=litellm_params.get("temperature", 0.0),
            max_tokens=litellm_params.get("max_tokens"),
            timeout=litellm_params.get("timeout"),
            num_retries=litellm_params.get("num_retries", 3),
        )

        print(f"✅ DeepEval LLM Manager: {self.model_name}")

    def get_llm(self) -> LiteLLMModel:
        """Get the configured DeepEval LiteLLM model."""
        return self.llm_model

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        return {
            "model_name": self.model_name,
            "temperature": self.litellm_params.get("temperature", 0.0),
            "max_tokens": self.litellm_params.get("max_tokens"),
            "timeout": self.litellm_params.get("timeout"),
            "num_retries": self.litellm_params.get("num_retries", 3),
        }
