"""Utility functions for evaluation processing."""

from typing import Optional

from .models import EvaluationDataConfig, EvaluationResult


def create_error_result(
    eval_config: EvaluationDataConfig,
    error_message: str,
    conversation_id: Optional[str] = None,
) -> EvaluationResult:
    """Create a standardized error result."""
    return EvaluationResult(
        eval_id=eval_config.eval_id,
        query=eval_config.eval_query,
        response="",
        eval_type=eval_config.eval_type,
        result="ERROR",
        conversation_group=eval_config.conversation_group,
        conversation_id=conversation_id,
        error=error_message,
    )


def create_success_result(
    eval_config: EvaluationDataConfig,
    response: str,
    success: bool,
    conversation_id: Optional[str] = None,
) -> EvaluationResult:
    """Create a standardized success/fail result."""
    return EvaluationResult(
        eval_id=eval_config.eval_id,
        query=eval_config.eval_query,
        response=response,
        eval_type=eval_config.eval_type,
        result="PASS" if success else "FAIL",
        conversation_group=eval_config.conversation_group,
        conversation_id=conversation_id,
        error=None,
    )
