"""Tests for keywords eval metric."""

from lightspeed_evaluation.core.metrics.custom.keywords_eval import evaluate_keywords
from lightspeed_evaluation.core.models import TurnData


class TestKeywordsEval:
    """Test cases for keywords eval metric."""

    def test_keywords_eval_first_list_all_matched(self):
        """Test successful keywords evaluation when first list has all keywords matched."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="This response contains openshift-monitoring and yes it exists",
            expected_keywords=[
                ["yes", "openshift-monitoring"],  # Option 1: Both keywords should match
                ["confirmed", "monitoring"],  # Option 2: Should not be checked
            ],
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 1.0
        assert "Keywords eval successful: Option 1" in reason
        assert "all keywords matched: 'yes', 'openshift-monitoring'" in reason

    def test_keywords_eval_first_list_fails_second_succeeds(self):
        """Test keywords evaluation when first list fails but second list succeeds."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="This response contains monitoring and confirmed status",
            expected_keywords=[
                [
                    "yes",
                    "openshift-monitoring",
                ],  # Option 1: "yes" missing, "openshift-monitoring" missing
                ["monitoring", "confirmed"],  # Option 2: Both should match
            ],
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 1.0
        assert "Keywords eval successful: Option 2" in reason
        assert "all keywords matched: 'monitoring', 'confirmed'" in reason

    def test_keywords_eval_all_lists_fail(self):
        """Test keywords evaluation when all lists fail."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="This response contains nothing relevant",
            expected_keywords=[
                ["yes", "openshift-monitoring"],  # Option 1: Both missing
                ["confirmed", "monitoring"],  # Option 2: Both missing
            ],
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 0.0
        assert "Keywords eval failed: All options failed" in reason
        assert (
            "Option 1: unmatched ['yes', 'openshift-monitoring'], matched [none]"
            in reason
        )
        assert (
            "Option 2: unmatched ['confirmed', 'monitoring'], matched [none]" in reason
        )

    def test_keywords_eval_partial_match_in_failed_list(self):
        """Test keywords evaluation with partial matches in failed lists."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="This response contains monitoring but no confirmation",
            expected_keywords=[
                ["yes", "confirmed"],  # Option 1: Both missing
                [
                    "monitoring",
                    "openshift",
                ],  # Option 2: "monitoring" matches, "openshift" missing
            ],
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 0.0
        assert "Keywords eval failed: All options failed" in reason
        assert "Option 1: unmatched ['yes', 'confirmed'], matched [none]" in reason
        assert "Option 2: unmatched ['openshift'], matched ['monitoring']" in reason

    def test_keywords_eval_case_insensitive(self):
        """Test that keywords evaluation is case insensitive."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="This response contains YES and OPENSHIFT-MONITORING",
            expected_keywords=[
                ["yes", "openshift-monitoring"]  # Should match despite case differences
            ],
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 1.0
        assert "Keywords eval successful: Option 1" in reason
        assert "all keywords matched: 'yes', 'openshift-monitoring'" in reason

    def test_keywords_eval_substring_matching(self):
        """Test that keywords evaluation works with substring matching."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="The openshift-monitoring-operator is running successfully",
            expected_keywords=[
                [
                    "monitoring",
                    "success",
                ]  # Should match "monitoring" in "openshift-monitoring-operator"
            ],
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 1.0
        assert "Keywords eval successful: Option 1" in reason
        assert "all keywords matched: 'monitoring', 'success'" in reason

    def test_keywords_eval_no_expected_keywords(self):
        """Test keywords evaluation when no expected keywords provided."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="Some response",
            expected_keywords=None,
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score is None
        assert "No expected keywords provided" in reason

    def test_keywords_eval_no_response(self):
        """Test keywords evaluation when no response provided."""
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response=None,
            expected_keywords=[["yes"], ["monitoring"]],
        )

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 0.0
        assert "No response provided" in reason

    def test_keywords_eval_empty_response(self):
        """Test keywords evaluation with empty response."""
        # Create turn data with valid response first, then modify it
        turn_data = TurnData(
            turn_id="test_turn",
            query="Test query",
            response="valid response",
            expected_keywords=[["yes"], ["monitoring"]],
        )
        # Manually set response to empty to bypass validation
        turn_data.response = ""

        score, reason = evaluate_keywords(None, 0, turn_data, False)

        assert score == 0.0
        assert "No response provided" in reason

    def test_keywords_eval_conversation_level_error(self):
        """Test that keywords_eval returns error for conversation-level evaluation."""
        score, reason = evaluate_keywords(None, None, None, True)

        assert score is None
        assert "Keywords eval is a turn-level metric" in reason

    def test_keywords_eval_no_turn_data(self):
        """Test keywords evaluation when no turn data provided."""
        score, reason = evaluate_keywords(None, 0, None, False)

        assert score is None
        assert "TurnData is required" in reason
