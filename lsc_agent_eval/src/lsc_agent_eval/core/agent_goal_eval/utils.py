"""Utility functions for evaluation processing."""

from typing import Optional, TypedDict

from .models import EvaluationDataConfig, EvaluationResult


class EvalResultItem(TypedDict):
    """Data model for result."""

    eval_type: str
    result: str
    error: Optional[str]


def create_evaluation_results(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    eval_config: EvaluationDataConfig,
    response: str = "",
    evaluation_results: Optional[list[EvalResultItem]] = None,
    error_message: Optional[str] = None,
    conversation_id: Optional[str] = None,
    tool_calls: Optional[list[list[dict]]] = None,
) -> list[EvaluationResult]:
    """Create standardized evaluation results."""
    results = []

    if error_message:
        # Global error - create ERROR results for all eval types
        for eval_type in eval_config.eval_types:
            results.append(
                EvaluationResult(
                    eval_id=eval_config.eval_id,
                    query=eval_config.eval_query,
                    response=response,
                    eval_type=eval_type,
                    result="ERROR",
                    conversation_group=eval_config.conversation_group,
                    conversation_id=conversation_id,
                    error=error_message,
                    tool_calls=(tool_calls if eval_type == "tool_eval" else None),
                    expected_intent=(
                        eval_config.expected_intent
                        if eval_type == "response_eval:intent"
                        else None
                    ),
                )
            )
    elif evaluation_results:
        # Individual evaluation results
        for eval_result in evaluation_results:
            results.append(
                EvaluationResult(
                    eval_id=eval_config.eval_id,
                    query=eval_config.eval_query,
                    response=response,
                    eval_type=eval_result["eval_type"],
                    result=eval_result["result"],
                    conversation_group=eval_config.conversation_group,
                    conversation_id=conversation_id,
                    error=eval_result["error"],
                    tool_calls=(
                        tool_calls if eval_result["eval_type"] == "tool_eval" else None
                    ),
                    expected_intent=(
                        eval_config.expected_intent
                        if eval_result["eval_type"] == "response_eval:intent"
                        else None
                    ),
                )
            )
    else:
        raise ValueError("Must provide either evaluation_results or error_message")

    return results
