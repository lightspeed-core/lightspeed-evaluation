"""DeepEval LLM Manager - DeepEval-specific LLM wrapper."""

from typing import Any

from deepeval.models import LiteLLMModel


class DeepEvalLLMManager:
    """DeepEval LLM Manager - Takes LLM parameters directly.

    This manager focuses solely on DeepEval-specific LLM integration.
    """

    def __init__(self, model_name: str, llm_params: dict[str, Any]):
        """Initialize with LLM parameters from LLMManager."""
        self.model_name = model_name
        self.llm_params = llm_params

        # Create DeepEval's LLM model with provided parameters
        self.llm_model = LiteLLMModel(
            model=self.model_name,
            temperature=llm_params.get("temperature", 0.0),
            max_tokens=llm_params.get("max_tokens"),
            timeout=llm_params.get("timeout"),
            num_retries=llm_params.get("num_retries", 3),
        )

        print(f"✅ DeepEval LLM Manager: {self.model_name}")

    def get_llm(self) -> LiteLLMModel:
        """Get the configured DeepEval LLM model."""
        return self.llm_model

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the configured model."""
        return {
            "model_name": self.model_name,
            "temperature": self.llm_params.get("temperature", 0.0),
            "max_tokens": self.llm_params.get("max_tokens"),
            "timeout": self.llm_params.get("timeout"),
            "num_retries": self.llm_params.get("num_retries", 3),
        }
