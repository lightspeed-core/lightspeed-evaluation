"""Base Custom LLM class for evaluation framework."""

from typing import Any, Optional, Union

import litellm

from lightspeed_evaluation.core.system.exceptions import LLMError


class BaseCustomLLM:  # pylint: disable=too-few-public-methods
    """Base LLM class with core calling functionality."""

    def __init__(self, model_name: str, llm_params: dict[str, Any]):
        """Initialize with model configuration."""
        self.model_name = model_name
        self.llm_params = llm_params

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
            "max_tokens": self.llm_params.get("max_tokens"),
            "timeout": self.llm_params.get("timeout"),
            "num_retries": self.llm_params.get("num_retries", 3),
            **kwargs,
        }

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

        except Exception as e:
            raise LLMError(f"LLM call failed: {str(e)}") from e
