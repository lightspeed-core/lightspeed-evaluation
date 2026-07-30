"""Mean Reciprocal Rank (MRR) evaluation for RAG retrieval quality.

Uses semantic similarity (sentence-transformers embeddings + cosine similarity)
for context matching with a configurable similarity threshold.

Falls back to normalized substring containment when sentence-transformers is not
installed.
"""

import logging
import re
from typing import Any, Optional

import numpy as np

from lightspeed_evaluation.core.models import TurnData

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.65


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _is_context_match_substring(retrieved: str, expected: str) -> bool:
    """Fallback substring containment match.

    Used when sentence-transformers is not available.
    """
    norm_retrieved = _normalize_text(retrieved)
    norm_expected = _normalize_text(expected)
    if not norm_retrieved or not norm_expected:
        return False
    return norm_expected in norm_retrieved or norm_retrieved in norm_expected


def _compute_similarity_matrix(
    retrieved_texts: list[str],
    expected_texts: list[str],
    model: Any,
) -> np.ndarray:
    """Compute cosine similarity matrix between retrieved and expected contexts.

    Args:
        retrieved_texts: Retrieved context strings.
        expected_texts: Expected (ground-truth) context strings.
        model: A SentenceTransformer model instance.

    Returns:
        Similarity matrix of shape ``(len(retrieved), len(expected))``.
    """
    ret_emb = model.encode(retrieved_texts, normalize_embeddings=True)
    exp_emb = model.encode(expected_texts, normalize_embeddings=True)
    return np.asarray(ret_emb) @ np.asarray(exp_emb).T


def _resolve_threshold(
    mrr_config: Optional[dict[str, Any]],
) -> float:
    """Determine the similarity threshold from config.

    Args:
        mrr_config: Metric metadata dict (may contain
            ``default_similarity_threshold``).

    Returns:
        The similarity threshold.
    """
    config = mrr_config or {}
    return float(
        config.get("default_similarity_threshold", DEFAULT_SIMILARITY_THRESHOLD)
    )


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
    *,
    embedding_model: Any = None,
    _embedding_model_name: str = "",
    mrr_config: Optional[dict[str, Any]] = None,
) -> tuple[Optional[float], str]:
    """Evaluate Mean Reciprocal Rank of retrieved contexts.

    When *embedding_model* is provided (a ``SentenceTransformer`` instance),
    matching uses cosine similarity with a configurable threshold.
    Otherwise falls back to substring containment.

    Args:
        _conv_data: Conversation data (unused).
        _turn_idx: Turn index (unused).
        turn_data: Turn data with ``contexts`` and ``expected_contexts``.
        is_conversation: Whether this is conversation-level evaluation.
        embedding_model: Optional SentenceTransformer model for semantic
            matching.
        embedding_model_name: Model identifier (unused in base, used by CRC).
        mrr_config: Optional metric metadata with keys such as
            ``default_similarity_threshold``.

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

    if embedding_model is not None:
        return _evaluate_mrr_semantic(
            turn_data.contexts,
            turn_data.expected_contexts,
            embedding_model,
            mrr_config,
        )

    return _evaluate_mrr_substring(turn_data.contexts, turn_data.expected_contexts)


def _evaluate_mrr_substring(
    contexts: list[str],
    expected_contexts: list[str],
) -> tuple[Optional[float], str]:
    """MRR via substring containment (fallback)."""
    for rank, retrieved in enumerate(contexts, start=1):
        for expected in expected_contexts:
            if _is_context_match_substring(retrieved, expected):
                score = 1.0 / rank
                return (
                    score,
                    f"MRR: {score:.4f} — first relevant context at rank {rank} "
                    f"(substring fallback)",
                )

    return (
        0.0,
        f"MRR: 0.0 — no relevant context found in "
        f"{len(contexts)} retrieved contexts (substring fallback)",
    )


def _evaluate_mrr_semantic(
    contexts: list[str],
    expected_contexts: list[str],
    model: Any,
    mrr_config: Optional[dict[str, Any]],
) -> tuple[Optional[float], str]:
    """MRR via semantic similarity with configurable threshold."""
    threshold = _resolve_threshold(mrr_config)
    sim_matrix = _compute_similarity_matrix(contexts, expected_contexts, model)

    for rank in range(len(contexts)):
        row_max = float(np.max(sim_matrix[rank]))
        if row_max >= threshold:
            score = 1.0 / (rank + 1)
            return (
                score,
                f"MRR: {score:.4f} — first relevant context at rank {rank + 1} "
                f"(similarity={row_max:.4f}, threshold={threshold:.4f})",
            )

    best_sim = float(np.max(sim_matrix))
    return (
        0.0,
        f"MRR: 0.0 — no context above threshold in "
        f"{len(contexts)} retrieved (best_similarity={best_sim:.4f}, "
        f"threshold={threshold:.4f})",
    )
