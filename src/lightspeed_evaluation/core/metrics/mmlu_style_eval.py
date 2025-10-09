"""MMLU-style multiple choice evaluation metrics."""

import re
from typing import Any, Optional

from lightspeed_evaluation.core.models import EvaluationScope, TurnData


class MultipleChoiceExactMatch:  # pylint: disable=too-few-public-methods
    """Exact match metric for multiple choice questions (MMLU-style scoring).

    Returns 1.0 for correct answer, 0.0 for incorrect.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        """Initialize metric.

        Args:
            threshold: Score threshold for passing (default: 1.0, meaning must be exact).
        """
        self.threshold = threshold

    def evaluate(  # pylint: disable=unused-argument
        self, response: str, expected_response: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Evaluate if the AI response matches the expected answer.

        Args:
            response: The AI's generated response.
            expected_response: The correct answer (e.g., "A", "B", "C", or "D").
            **kwargs: Additional arguments (ignored).

        Returns:
            Dict with 'score' (1.0 or 0.0) and 'reason' (explanation).
        """
        # Clean inputs
        response_clean = response.strip().upper()
        expected_clean = expected_response.strip().upper()

        # Extract letter from response using regex
        # Handles cases like:
        # - "B"
        # - "The answer is B"
        # - "B) Code can survive..."
        # - "I think B is correct"
        letter_match = re.search(r"\b([ABCD])\b", response_clean)

        if letter_match:
            response_letter = letter_match.group(1)
        else:
            # No clear letter found, try first character
            response_letter = response_clean[0] if response_clean else ""

        # Compare
        is_correct = response_letter == expected_clean
        score = 1.0 if is_correct else 0.0

        # Build explanation
        reason = (
            f"Expected: {expected_clean} | "
            f"Extracted: {response_letter} | "
            f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'} | "
            f"Full response: '{response[:100]}...'"
            if len(response) > 100
            else f"Full response: '{response}'"
        )

        return {"score": score, "reason": reason}


class MultipleChoiceStrictMatch:  # pylint: disable=too-few-public-methods
    """Stricter version requiring response to be exactly A, B, C, or D."""

    def __init__(self, threshold: float = 1.0) -> None:
        """Initialize metric.

        Args:
            threshold: Score threshold for passing (default: 1.0).
        """
        self.threshold = threshold

    def evaluate(  # pylint: disable=unused-argument
        self, response: str, expected_response: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Evaluate if response exactly matches expected answer.

        Args:
            response: The AI's generated response.
            expected_response: The correct answer (single letter).
            **kwargs: Additional arguments (ignored).

        Returns:
            Dict with 'score' (1.0 or 0.0) and 'reason' (explanation).
        """
        response_clean = response.strip().upper()
        expected_clean = expected_response.strip().upper()

        # Must be exactly one letter
        is_correct = response_clean == expected_clean and len(response_clean) == 1
        score = 1.0 if is_correct else 0.0

        return {
            "score": score,
            "reason": f"Expected exactly '{expected_clean}', got '{response_clean}'",
        }


class MMLUMetrics:  # pylint: disable=too-few-public-methods
    """Custom MMLU-style metrics integrated with the evaluation framework."""

    def __init__(self) -> None:
        """Initialize MMLU metrics."""
        self.exact_match = MultipleChoiceExactMatch()
        self.strict_match = MultipleChoiceStrictMatch()

        self.supported_metrics = {
            "mmlu_exact_match": self._evaluate_exact_match,
            "mmlu_strict_match": self._evaluate_strict_match,
        }

    def evaluate(
        self,
        metric_name: str,
        conv_data: Any,
        scope: EvaluationScope,
    ) -> tuple[Optional[float], str]:
        """Evaluate an MMLU-style metric.

        Args:
            metric_name: Name of the metric to evaluate.
            conv_data: Conversation data (unused for MMLU metrics).
            scope: Evaluation scope containing turn data.

        Returns:
            Tuple of (score, reason) where score is between 0.0 and 1.0.
        """
        if metric_name not in self.supported_metrics:
            return None, f"Unsupported MMLU metric: {metric_name}"

        try:
            return self.supported_metrics[metric_name](
                conv_data, scope.turn_idx, scope.turn_data, scope.is_conversation
            )
        except (ValueError, AttributeError, KeyError) as e:
            return None, f"MMLU {metric_name} evaluation failed: {str(e)}"

    def _evaluate_exact_match(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate using exact match with flexible letter extraction.

        Args:
            _conv_data: Conversation data (unused).
            _turn_idx: Turn index (unused).
            turn_data: Turn data containing response and expected response.
            is_conversation: Whether this is conversation-level evaluation.

        Returns:
            Tuple of (score, reason).
        """
        if is_conversation:
            return None, "MMLU exact match is a turn-level metric"

        if turn_data is None:
            return None, "TurnData is required for MMLU evaluation"

        if not turn_data.response:
            return None, "Response is required for MMLU evaluation"

        if not turn_data.expected_response:
            return None, "Expected response is required for MMLU evaluation"

        result = self.exact_match.evaluate(
            turn_data.response, turn_data.expected_response
        )
        return result["score"], result["reason"]

    def _evaluate_strict_match(
        self,
        _conv_data: Any,
        _turn_idx: Optional[int],
        turn_data: Optional[TurnData],
        is_conversation: bool,
    ) -> tuple[Optional[float], str]:
        """Evaluate using strict exact match (single letter only).

        Args:
            _conv_data: Conversation data (unused).
            _turn_idx: Turn index (unused).
            turn_data: Turn data containing response and expected response.
            is_conversation: Whether this is conversation-level evaluation.

        Returns:
            Tuple of (score, reason).
        """
        if is_conversation:
            return None, "MMLU strict match is a turn-level metric"

        if turn_data is None:
            return None, "TurnData is required for MMLU evaluation"

        if not turn_data.response:
            return None, "Response is required for MMLU evaluation"

        if not turn_data.expected_response:
            return None, "Expected response is required for MMLU evaluation"

        result = self.strict_match.evaluate(
            turn_data.response, turn_data.expected_response
        )
        return result["score"], result["reason"]
