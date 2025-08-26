"""Ragas LLM Manager - Ragas-specific LLM wrapper that takes LiteLLM parameters."""

from typing import Any, Dict, List, Optional

import litellm
from ragas.llms.base import BaseRagasLLM, Generation, LLMResult
from ragas.metrics import answer_relevancy, faithfulness


class RagasCustomLLM(BaseRagasLLM):
    """Custom LLM for Ragas using LiteLLM parameters."""

    def __init__(self, model_name: str, litellm_params: Dict[str, Any]):
        """Initialize Ragas custom LLM with model name and LiteLLM parameters."""
        super().__init__()
        self.model_name = model_name
        self.litellm_params = litellm_params
        print(f"✅ Ragas Custom LLM: {self.model_name}")

    def generate_text(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        prompt: Any,
        n: int = 1,
        temperature: float = 1e-08,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
    ) -> LLMResult:
        """Generate text using LiteLLM with provided parameters."""
        prompt_text = str(prompt)

        # Use temperature from params unless explicitly overridden
        temp = (
            temperature
            if temperature != 1e-08
            else self.litellm_params.get("temperature", 0.0)
        )

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                n=n,
                temperature=temp,
                max_tokens=self.litellm_params.get("max_tokens"),
                timeout=self.litellm_params.get("timeout"),
                num_retries=self.litellm_params.get("num_retries"),
            )

            # Convert to Ragas format
            generations = []
            for choice in response.choices:  # type: ignore
                content = choice.message.content  # type: ignore
                if content is None:
                    content = ""
                gen = Generation(text=content.strip())
                generations.append(gen)

            result = LLMResult(generations=[generations])
            return result

        except Exception as e:
            print(f"❌ Ragas LLM failed: {e}")
            raise RuntimeError(f"Ragas LLM evaluation failed: {str(e)}") from e

    async def agenerate_text(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        prompt: Any,
        n: int = 1,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
    ) -> LLMResult:
        """Async generate."""
        temp = temperature if temperature is not None else 1e-08
        return self.generate_text(
            prompt, n=n, temperature=temp, stop=stop, callbacks=callbacks
        )

    def is_finished(self, response: LLMResult) -> bool:
        """Check if response is complete."""
        return True


class RagasLLMManager:
    """
    Ragas LLM Manager - Takes LLM parameters directly.

    This manager focuses solely on Ragas-specific LLM integration.
    """

    def __init__(self, model_name: str, litellm_params: Dict[str, Any]):
        """Initialize with LLM parameters from LLMManager."""
        self.model_name = model_name
        self.litellm_params = litellm_params
        self.custom_llm = RagasCustomLLM(model_name, litellm_params)

        # Configure Ragas metrics to use our custom LLM
        answer_relevancy.llm = self.custom_llm
        faithfulness.llm = self.custom_llm

        print("✅ Ragas LLM Manager configured")

    def get_llm(self) -> RagasCustomLLM:
        """Get the configured Ragas LLM."""
        return self.custom_llm

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model."""
        return {
            "model_name": self.model_name,
            "temperature": self.litellm_params.get("temperature", 0.0),
        }
