"""Utility functions for evaluation processing."""

from typing import Optional

from .models import EvaluationDataConfig, EvaluationResult


def create_evaluation_results(
    eval_config: EvaluationDataConfig,
    response: str = "",
    evaluation_results: Optional[list[dict]] = None,
    error_message: Optional[str] = None,
    conversation_id: Optional[str] = None,
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
                )
            )
    else:
        raise ValueError("Must provide either evaluation_results or error_message")

    return results
