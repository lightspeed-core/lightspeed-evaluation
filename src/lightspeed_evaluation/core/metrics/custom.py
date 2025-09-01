"""Custom metrics using direct LLM integration."""

import re
from typing import Any, Dict, Optional, Tuple

import litellm
from pydantic import BaseModel, Field

from ..config.models import TurnData
from ..llm.manager import LLMManager
from ..output.statistics import EvaluationScope


class EvaluationPromptParams(BaseModel):
    """Parameters for evaluation prompt creation."""

    metric_name: str = Field(..., description="Name of the metric being evaluated")
    query: str = Field(..., description="The user query")
    response: str = Field(..., description="The model response")
    expected_response: Optional[str] = Field(
        None, description="Expected response if available"
    )
    contexts: Optional[list] = Field(
        None, description="Context information if available"
    )
    scale: str = Field("0.0 to 1.0", description="Scale for scoring")


class CustomMetrics:
    """Handles custom metrics using LLMManager for direct LiteLLM calls."""

    def __init__(self, llm_manager: LLMManager):
        """
        Initialize with LLM Manager.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters
        """
        self.model_name = llm_manager.get_model_name()
        self.litellm_params = llm_manager.get_litellm_params()

        self.supported_metrics = {
            "answer_correctness": self._evaluate_answer_correctness
        }

        print(f"✅ Custom Metrics initialized: {self.model_name}")

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        scope: EvaluationScope,
    ) -> Tuple[Optional[float], str]:
        """Evaluate a custom metric."""
        if metric_name not in self.supported_metrics:
            return None, f"Unsupported custom metric: {metric_name}"

        try:
            return self.supported_metrics[metric_name](
                conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
            )
        except (ValueError, AttributeError, KeyError) as e:
            return None, f"Custom {metric_name} evaluation failed: {str(e)}"

    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Make a LiteLLM call with the configured parameters."""
        # Prepare messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                temperature=self.litellm_params.get("temperature", 0.0),
                max_tokens=self.litellm_params.get("max_tokens"),
                timeout=self.litellm_params.get("timeout"),
                num_retries=self.litellm_params.get("num_retries", 3),
            )

            content = response.choices[0].message.content  # type: ignore
            if content is None:
                raise RuntimeError("LLM returned empty response")
            return content.strip()

        except Exception as e:
            raise RuntimeError(f"LiteLLM call failed: {str(e)}") from e

    def _parse_score_response(self, response: str) -> Tuple[Optional[float], str]:
        r"""
        Parse LLM response to extract score and reason.

        Expected formats:
        - "Score: 0.85\nReason: The answer is accurate..."
        - "8.5/10 - The response is comprehensive..."
        - "Rating: 4.2 out of 5"
        """
        lines = response.split("\n")
        score = None
        reason = response  # Default to full response

        # Try to find explicit score/reason format
        for line in lines:
            line = line.strip()
            if line.lower().startswith("score:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    score = float(score_text)
                except (ValueError, IndexError):
                    pass
            elif line.lower().startswith("reason:"):
                try:
                    reason = line.split(":", 1)[1].strip()
                except IndexError:
                    pass

        # If no explicit score found, try to extract from text
        if score is None:
            score = self._extract_score_from_text(response)

        # Normalize score to 0-1 range if needed
        if score is not None and score > 1.0:
            if score <= 10.0:  # Assume 0-10 scale
                score = score / 10.0
            elif score <= 100.0:  # Assume 0-100 scale
                score = score / 100.0

        return score, reason

    def _extract_score_from_text(self, text: str) -> Optional[float]:
        """Extract numeric score from text using various patterns."""
        # Pattern 1: "X.Y/Z" format (e.g., "8.5/10", "4.2/5")
        fraction_pattern = r"(\d+\.?\d*)/(\d+\.?\d*)"
        fraction_match = re.search(fraction_pattern, text)
        if fraction_match:
            numerator = float(fraction_match.group(1))
            denominator = float(fraction_match.group(2))
            return numerator / denominator if denominator > 0 else None

        # Pattern 2: "X out of Y" format
        out_of_pattern = r"(\d+\.?\d*)\s+out\s+of\s+(\d+\.?\d*)"
        out_of_match = re.search(out_of_pattern, text, re.IGNORECASE)
        if out_of_match:
            numerator = float(out_of_match.group(1))
            denominator = float(out_of_match.group(2))
            return numerator / denominator if denominator > 0 else None

        # Pattern 3: First decimal number found
        decimal_pattern = r"\d+\.\d+"
        decimal_match = re.search(decimal_pattern, text)
        if decimal_match:
            return float(decimal_match.group())

        # Pattern 4: First integer found
        integer_pattern = r"\d+"
        integer_match = re.search(integer_pattern, text)
        if integer_match:
            return float(integer_match.group())

        return None

    def _create_evaluation_prompt(self, params: EvaluationPromptParams) -> str:
        """Create a standardized evaluation prompt for custom metrics."""
        prompt_parts = [
            f"Evaluate the {params.metric_name} of the given response "
            f"on a scale of {params.scale}.",
            "",
            f"Question: {params.query}",
            f"Response: {params.response}",
        ]

        if params.expected_response:
            prompt_parts.append(f"Expected Response: {params.expected_response}")

        if params.contexts:
            prompt_parts.append("Context:")
            for i, ctx in enumerate(params.contexts, 1):
                if isinstance(ctx, dict):
                    content = ctx.get("content", str(ctx))
                else:
                    content = str(ctx)
                prompt_parts.append(f"{i}. {content}")

        prompt_parts.extend(
            [
                "",
                f"Rate the {params.metric_name} and provide your reasoning.",
                "",
                "Format your response as:",
                f"Score: [your score on {params.scale}]",
                "Reason: [your detailed explanation]",
            ]
        )

        return "\n".join(prompt_parts)

    def _evaluate_answer_correctness(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> Tuple[Optional[float], str]:
        """Evaluate answer correctness using custom prompt."""
        if is_conversation:
            return None, "Answer correctness is a turn-level metric"

        if turn_data is None:
            return None, "TurnData is required for answer correctness evaluation"

        query = turn_data.query
        response = turn_data.response
        expected_response = turn_data.expected_response

        # Create evaluation prompt
        params = EvaluationPromptParams(
            metric_name="answer correctness",
            query=query,
            response=response,
            expected_response=expected_response,
            contexts=turn_data.contexts if turn_data.contexts else None,
            scale="0.0 to 1.0",
        )
        prompt = self._create_evaluation_prompt(params)

        # Add specific instructions for answer correctness
        prompt += "\n\nConsider:\n"
        prompt += "- Factual accuracy compared to expected answer\n"
        prompt += "- Completeness of information\n"
        prompt += "- Alignment with expected response\n"
        prompt += "- Absence of contradictory information"

        # Make LLM call and parse response
        llm_response = self._call_llm(prompt)
        score, reason = self._parse_score_response(llm_response)

        if score is None:
            return (
                None,
                f"Could not parse score from LLM response: {llm_response[:100]}...",
            )

        return score, f"Custom answer correctness: {score:.2f} - {reason}"

    @classmethod
    def from_system_config(cls, system_config: Dict[str, Any]) -> "CustomMetrics":
        """Create CustomMetrics from system configuration."""
        llm_manager = LLMManager.from_system_config(system_config)
        return cls(llm_manager)
