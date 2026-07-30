"""Mean Reciprocal Rank (MRR) evaluation for RAG retrieval quality."""

import re
from typing import Any, Optional

from lightspeed_evaluation.core.models import TurnData


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _is_context_match(retrieved: str, expected: str) -> bool:
    """Check if a retrieved context matches an expected context.

    Uses normalized containment: a match occurs when either text
    contains the other after normalization.  This handles retriever
    chunks that are supersets or subsets of the ground-truth context.
    """
    norm_retrieved = _normalize_text(retrieved)
    norm_expected = _normalize_text(expected)
    if not norm_retrieved or not norm_expected:
        return False
    return norm_expected in norm_retrieved or norm_retrieved in norm_expected


def _validate_inputs(
    is_conversation: bool, turn_data: Optional[TurnData]
) -> Optional[tuple[Optional[float], str]]:
    """Validate inputs for MRR evaluation."""
    if is_conversation:
        return None, "MRR is a turn-level metric"

    if turn_data is None:
        return None, "TurnData is required for MRR evaluation"

    if not turn_data.contexts:
        return None, "Retrieved contexts are required for MRR evaluation"

    if not turn_data.expected_contexts:
        return None, "Expected contexts are required for MRR evaluation"

    return None


def evaluate_mrr(
    _conv_data: Any,
    _turn_idx: Optional[int],
    turn_data: Optional[TurnData],
    is_conversation: bool,
) -> tuple[Optional[float], str]:
    """Evaluate Mean Reciprocal Rank of retrieved contexts.

    MRR measures how high the first relevant context appears in the
    ranked list of retrieved contexts.  The score is 1/rank for the
    first match, or 0.0 when no retrieved context matches any expected
    context.

    Args:
        _conv_data: Conversation data (unused).
        _turn_idx: Turn index (unused).
        turn_data: Turn data with ``contexts`` and ``expected_contexts``.
        is_conversation: Whether this is conversation-level evaluation.

    Returns:
        Tuple of (score, reason).
    """
    validation_result = _validate_inputs(is_conversation, turn_data)
    if validation_result:
        return validation_result

    if (
        turn_data is None
        or turn_data.contexts is None
        or turn_data.expected_contexts is None
    ):
        return None, "Invalid turn data after validation"

    for rank, retrieved in enumerate(turn_data.contexts, start=1):
        for expected in turn_data.expected_contexts:
            if _is_context_match(retrieved, expected):
                score = 1.0 / rank
                return (
                    score,
                    f"MRR: {score:.4f} — first relevant context at rank {rank}",
                )

    return (
        0.0,
        f"MRR: 0.0 — no relevant context found in "
        f"{len(turn_data.contexts)} retrieved contexts",
    )
