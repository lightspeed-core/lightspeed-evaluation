"""Tests for MRR (Mean Reciprocal Rank) evaluation metric."""

import pytest

from lightspeed_evaluation.core.metrics.custom.mrr_eval import (
    _is_context_match,
    _normalize_text,
    evaluate_mrr,
)
from lightspeed_evaluation.core.models import TurnData


class TestNormalizeText:
    """Test text normalization helper."""

    def test_lowercase(self) -> None:
        """Test that text is lowercased."""
        assert _normalize_text("Hello World") == "hello world"

    def test_collapse_whitespace(self) -> None:
        """Test that multiple whitespace characters are collapsed."""
        assert _normalize_text("hello   world") == "hello world"

    def test_strip_whitespace(self) -> None:
        """Test that leading and trailing whitespace is stripped."""
        assert _normalize_text("  hello world  ") == "hello world"

    def test_mixed_whitespace(self) -> None:
        """Test tabs, newlines, and multiple spaces."""
        assert _normalize_text("hello\t\n  world") == "hello world"


class TestIsContextMatch:
    """Test context matching logic."""

    def test_exact_match(self) -> None:
        """Test that identical strings match."""
        assert _is_context_match("hello world", "hello world") is True

    def test_case_insensitive(self) -> None:
        """Test that matching is case-insensitive."""
        assert _is_context_match("Hello World", "hello world") is True

    def test_retrieved_contains_expected(self) -> None:
        """Test match when retrieved context is a superset of expected."""
        retrieved = (
            "Introduction: RHEL is a Linux distribution. It is enterprise-grade."
        )
        expected = "RHEL is a Linux distribution"
        assert _is_context_match(retrieved, expected) is True

    def test_expected_contains_retrieved(self) -> None:
        """Test match when expected context is a superset of retrieved."""
        retrieved = "RHEL is a Linux distribution"
        expected = "Introduction: RHEL is a Linux distribution. It is enterprise-grade."
        assert _is_context_match(retrieved, expected) is True

    def test_no_match(self) -> None:
        """Test that unrelated strings do not match."""
        assert _is_context_match("hello world", "goodbye moon") is False

    def test_whitespace_normalization(self) -> None:
        """Test match with different whitespace."""
        assert _is_context_match("hello   world", "hello world") is True


class TestEvaluateMrr:
    """Test the main evaluate_mrr function."""

    def test_first_context_matches(self) -> None:
        """Test score is 1.0 when first retrieved context is relevant."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=[
                "RHEL is a Linux distribution",
                "Ubuntu is another distribution",
            ],
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 1.0
        assert "rank 1" in reason

    def test_second_context_matches(self) -> None:
        """Test score is 0.5 when second retrieved context is relevant."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=[
                "Ubuntu is a distribution",
                "RHEL is a Linux distribution",
                "Fedora is upstream",
            ],
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 0.5
        assert "rank 2" in reason

    def test_third_context_matches(self) -> None:
        """Test score is 1/3 when third retrieved context is relevant."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=[
                "Ubuntu is a distribution",
                "Fedora is upstream",
                "RHEL is a Linux distribution",
            ],
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == pytest.approx(1.0 / 3)
        assert "rank 3" in reason

    def test_no_match_returns_zero(self) -> None:
        """Test score is 0.0 when no retrieved context is relevant."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=[
                "Ubuntu is a distribution",
                "Fedora is upstream",
            ],
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 0.0
        assert "no relevant context found" in reason

    def test_case_insensitive_matching(self) -> None:
        """Test that matching is case-insensitive."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=["rhel IS a LINUX Distribution"],
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, _reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 1.0

    def test_whitespace_normalized_matching(self) -> None:
        """Test that whitespace differences are normalized."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=["RHEL  is\ta   Linux\ndistribution"],
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, _reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 1.0

    def test_retrieved_superset_of_expected(self) -> None:
        """Test match when retrieved context contains the expected text."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=[
                "Introduction: RHEL is a Linux distribution. It is enterprise-grade.",
            ],
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, _reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 1.0

    def test_expected_superset_of_retrieved(self) -> None:
        """Test match when expected context contains the retrieved text."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=["RHEL is a Linux distribution"],
            expected_contexts=[
                "Introduction: RHEL is a Linux distribution. It is enterprise-grade.",
            ],
        )

        score, _reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 1.0

    def test_multiple_expected_contexts(self) -> None:
        """Test with multiple expected contexts where second one matches."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=[
                "Unrelated text",
                "Fedora is the upstream project for RHEL",
            ],
            expected_contexts=[
                "RHEL is a Linux distribution",
                "Fedora is the upstream project for RHEL",
            ],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 0.5
        assert "rank 2" in reason

    def test_conversation_level_returns_none(self) -> None:
        """Test that conversation-level evaluation returns None."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=["some context"],
            expected_contexts=["some context"],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, True)

        assert score is None
        assert "turn-level metric" in reason

    def test_missing_turn_data_returns_none(self) -> None:
        """Test that missing turn data returns None."""
        score, reason = evaluate_mrr(None, 0, None, False)

        assert score is None
        assert "TurnData is required" in reason

    def test_missing_contexts_returns_none(self) -> None:
        """Test that missing retrieved contexts returns None."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            expected_contexts=["RHEL is a Linux distribution"],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score is None
        assert "Retrieved contexts are required" in reason

    def test_missing_expected_contexts_returns_none(self) -> None:
        """Test that missing expected contexts returns None."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=["RHEL is a Linux distribution"],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score is None
        assert "Expected contexts are required" in reason

    def test_empty_contexts_rejected_by_model(self) -> None:
        """Test that empty contexts list is rejected by TurnData validation."""
        with pytest.raises(Exception):
            TurnData(
                turn_id="t1",
                query="What is RHEL?",
                contexts=[],
                expected_contexts=["RHEL is a Linux distribution"],
            )

    def test_empty_expected_contexts_returns_none(self) -> None:
        """Test that empty expected contexts list returns None."""
        turn_data = TurnData(
            turn_id="t1",
            query="What is RHEL?",
            contexts=["RHEL is a Linux distribution"],
            expected_contexts=[],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score is None
        assert "Expected contexts are required" in reason

    def test_first_expected_context_matches_first(self) -> None:
        """Test that earliest rank wins when multiple expected match different ranks."""
        turn_data = TurnData(
            turn_id="t1",
            query="Tell me about Linux",
            contexts=[
                "RHEL is enterprise Linux",
                "Ubuntu is desktop Linux",
                "Fedora is community Linux",
            ],
            expected_contexts=[
                "Ubuntu is desktop Linux",
                "RHEL is enterprise Linux",
            ],
        )

        score, reason = evaluate_mrr(None, 0, turn_data, False)

        assert score == 1.0
        assert "rank 1" in reason
