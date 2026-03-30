"""Base Custom LLM class for evaluation framework."""

import logging
from typing import Any, Union

import litellm
from litellm.exceptions import InternalServerError

from lightspeed_evaluation.core.llm.litellm_patch import setup_litellm_ssl
from lightspeed_evaluation.core.system.exceptions import LLMError

logger = logging.getLogger(__name__)


class BaseCustomLLM:  # pylint: disable=too-few-public-methods
    """Base LLM class with core calling functionality."""

    def __init__(self, model_name: str, llm_params: dict[str, Any]):
        """Initialize with model configuration.

        Args:
            model_name: Constructed model name (e.g., "openai/gpt-4")
            llm_params: LLM params from get_llm_params() — operational fields
                plus a "parameters" dict with inference params.
        """
        self.model_name = model_name
        self.llm_params = llm_params

        # Always drop unsupported parameters for cross-provider compatibility
        litellm.drop_params = True

        setup_litellm_ssl(llm_params)

    def call(
        self,
        prompt: str,
        n: int = 1,
        return_single: bool = True,
        **kwargs: Any,
    ) -> Union[str, list[str]]:
        """Make LLM call and return response(s).

        Args:
            prompt: Text prompt to send
            n: Number of responses to generate (default 1)
            return_single: If True and n=1, return single string. If False, always return list.
            **kwargs: Additional LLM parameters

        Returns:
            Single string if return_single=True and n=1, otherwise list of strings
        """
        # Note: Forbidden keys are rejected at LLMParametersConfig load time
        call_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "n": n,
            "timeout": self.llm_params.get("timeout"),
            "num_retries": self.llm_params.get("num_retries"),
            **self.llm_params.get("parameters", {}),
            **kwargs,
        }

        response = None
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

        except InternalServerError as e:
            # Check if it's an SSL/certificate error
            error_msg = str(e)
            if "[X509]" in error_msg or "PEM lib" in error_msg:
                raise LLMError(
                    f"Judge LLM SSL certificate verification failed: {error_msg}"
                ) from e

            # Otherwise, it's a different internal server error
            raise LLMError(f"LLM internal server error: {error_msg}") from e

        except Exception as e:
            raise LLMError(f"LLM call failed: {str(e)}") from e
