"""Tests for data models, specifically TurnData expected_tool_calls validation."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.data import TurnData


class TestTurnDataToolCallsValidation:
    """Test cases for TurnData expected_tool_calls field validation and conversion."""

    def test_single_set_format_conversion(self):
        """Test that single set format is converted to multiple sets format."""
        # Single set format (backward compatibility)
        turn_data = TurnData(
            turn_id="test_single",
            query="Test query",
            expected_tool_calls=[
                [{"tool_name": "test_tool", "arguments": {"key": "value"}}]
            ],
        )

        # Should be converted to multiple sets format
        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert expected is not None
        assert len(expected) == 1  # One alternative set
        assert len(expected[0]) == 1  # One sequence in the set
        assert len(expected[0][0]) == 1  # One tool call in the sequence
        assert expected[0][0][0]["tool_name"] == "test_tool"

    def test_multiple_sets_format_preserved(self):
        """Test that multiple sets format is preserved as-is."""
        # Multiple sets format
        turn_data = TurnData(
            turn_id="test_multiple",
            query="Test query",
            expected_tool_calls=[
                [  # First alternative set
                    [{"tool_name": "tool1", "arguments": {"key": "value1"}}]
                ],
                [  # Second alternative set
                    [{"tool_name": "tool2", "arguments": {"key": "value2"}}]
                ],
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert expected is not None
        assert len(expected) == 2  # Two alternative sets
        assert expected[0][0][0]["tool_name"] == "tool1"
        assert expected[1][0][0]["tool_name"] == "tool2"

    def test_empty_alternatives_allowed(self):
        """Test that empty alternatives are now allowed."""
        # This should be accepted (no longer rejected)
        turn_data = TurnData(
            turn_id="test_flexible",
            query="Test query",
            expected_tool_calls=[
                [  # Primary: use tool
                    [{"tool_name": "cache_check", "arguments": {"key": "data"}}]
                ],
                [],  # Alternative: skip tool (empty)
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 2
        assert len(expected[0]) == 1  # First set has one sequence
        assert len(expected[1]) == 0  # Second set is empty

    def test_multiple_tools_plus_empty_allowed(self):
        """Test that multiple tools + empty alternatives are allowed."""
        # This should be accepted with the new flexible approach
        turn_data = TurnData(
            turn_id="test_multiple_plus_empty",
            query="Test query",
            expected_tool_calls=[
                [  # Option 1: Use cache1
                    [{"tool_name": "cache1", "arguments": {"key": "data"}}]
                ],
                [  # Option 2: Use cache2
                    [{"tool_name": "cache2", "arguments": {"key": "data"}}]
                ],
                [],  # Option 3: Skip all (empty)
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 3
        assert len(expected[0]) == 1  # First set has tool
        assert len(expected[1]) == 1  # Second set has tool
        assert len(expected[2]) == 0  # Third set is empty

    def test_complex_sequences(self):
        """Test complex tool call sequences."""
        turn_data = TurnData(
            turn_id="test_complex",
            query="Test query",
            expected_tool_calls=[
                [  # Full sequence
                    [{"tool_name": "validate", "arguments": {}}],
                    [{"tool_name": "deploy", "arguments": {}}],
                ],
                [[{"tool_name": "deploy", "arguments": {}}]],  # Direct deploy
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 2
        assert len(expected[0]) == 2  # Two sequences in first set
        assert len(expected[1]) == 1  # One sequence in second set

    def test_empty_list_handling(self):
        """Test handling of empty lists at different levels."""
        # Valid: Non-empty alternative followed by empty alternatives
        turn_data = TurnData(
            turn_id="test_valid_empty",
            query="Test query",
            expected_tool_calls=[
                [[{"tool_name": "test_tool", "arguments": {}}]],  # Primary: non-empty
                [],  # Alternative: empty (valid as fallback)
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 2
        assert len(expected[0]) == 1  # First set has tool
        assert len(expected[1]) == 0  # Second set is empty

    def test_invalid_format_validation(self):
        """Test validation of invalid formats."""
        # Non-list format should be rejected
        with pytest.raises(ValidationError):
            TurnData(
                turn_id="test_invalid",
                query="Test query",
                expected_tool_calls="not_a_list",
            )

    def test_invalid_tool_call_structure(self):
        """Test validation of invalid tool call structure."""
        # Invalid tool call structure should be rejected
        with pytest.raises(ValidationError):
            TurnData(
                turn_id="test_invalid_structure",
                query="Test query",
                expected_tool_calls=[[[{"invalid": "structure"}]]],  # Missing tool_name
            )

    def test_none_expected_tool_calls(self):
        """Test that None is handled correctly."""
        turn_data = TurnData(
            turn_id="test_none", query="Test query", expected_tool_calls=None
        )

        assert turn_data.expected_tool_calls is None

    def test_regex_arguments_preserved(self):
        """Test that regex patterns in arguments are preserved."""
        turn_data = TurnData(
            turn_id="test_regex",
            query="Test query",
            expected_tool_calls=[
                [[{"tool_name": "get_pod", "arguments": {"name": "web-server-[0-9]+"}}]]
            ],
        )

        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert expected[0][0][0]["arguments"]["name"] == "web-server-[0-9]+"

    def test_empty_sequence_rejected(self):
        """Test that empty sequences are rejected (align with real API behavior)."""
        # [[]] is now detected as old format with empty sequence, which is invalid
        with pytest.raises(
            ValidationError,
            match="Empty sequence at position 0 in alternative 0 is invalid",
        ):
            TurnData(
                turn_id="test_invalid_empty_sequence",
                query="Test query",
                expected_tool_calls=[[]],  # Empty sequence - should be rejected
            )

    def test_empty_set_as_first_element_rejected(self):
        """Test that empty set as the first element is rejected."""
        # This is now detected as old format and empty sequence validation catches it
        with pytest.raises(
            ValidationError,
            match="Empty sequence at position 0 in alternative 0 is invalid",
        ):
            TurnData(
                turn_id="test_invalid_first_empty",
                query="Test query",
                expected_tool_calls=[
                    [],  # Empty sequence in old format - should be rejected
                    [[{"tool_name": "tool1", "arguments": {}}]],
                ],
            )

    def test_multiple_empty_sequences_rejected(self):
        """Test that multiple empty sequences are rejected (align with real API behavior)."""
        # [[], []] is now treated as multiple sets format with empty alternatives as first elements
        # This hits empty set constraints instead of empty sequence validation
        with pytest.raises(
            ValidationError, match="Empty set cannot be the first alternative"
        ):
            TurnData(
                turn_id="test_empty_sequences",
                query="Test query",
                expected_tool_calls=[
                    [],
                    [],
                ],  # Empty alternatives starting first - should be rejected
            )

    def test_mixed_empty_sequences_rejected(self):
        """Test that mixed empty/non-empty sequences are rejected."""
        # This scenario is impossible in real APIs
        with pytest.raises(
            ValidationError,
            match="Empty sequence at position 1 in alternative 0 is invalid",
        ):
            TurnData(
                turn_id="test_mixed_sequences",
                query="Test query",
                expected_tool_calls=[
                    [{"tool_name": "search", "arguments": {}}],  # Valid sequence
                    [],  # Empty sequence - should be rejected
                ],
            )

    def test_real_api_aligned_scenarios_accepted(self):
        """Test that scenarios matching real API behavior are accepted."""
        # These match actual API response structures
        valid_cases = [
            # Single tool sequence
            [[{"tool_name": "search", "arguments": {}}]],
            # Multiple tools in one sequence
            [
                [
                    {"tool_name": "search", "arguments": {}},
                    {"tool_name": "analyze", "arguments": {}},
                ]
            ],
            # Multiple sequences
            [
                [{"tool_name": "search", "arguments": {}}],
                [{"tool_name": "analyze", "arguments": {}}],
            ],
        ]

        for i, expected_tool_calls in enumerate(valid_cases):
            turn_data = TurnData(
                turn_id=f"test_valid_{i}",
                query="Test query",
                expected_tool_calls=expected_tool_calls,
            )
            # Should be accepted without errors
            assert turn_data.expected_tool_calls is not None

    def test_multiple_empty_alternatives_rejected(self):
        """Test that multiple empty alternatives are rejected as redundant."""
        with pytest.raises(
            ValidationError, match="Found 2 empty alternatives.*redundant"
        ):
            TurnData(
                turn_id="test_redundant_empty",
                query="Test query",
                expected_tool_calls=[
                    [[{"tool_name": "tool1", "arguments": {}}]],
                    [],  # Empty alternative 1
                    [],  # Empty alternative 2 (redundant!)
                ],
            )

    def test_actual_empty_sequences_rejected(self):
        """Test that actual empty sequences within alternatives are rejected."""
        # This tests the real empty sequence validation (not empty alternatives)
        with pytest.raises(
            ValidationError,
            match="Empty sequence at position 1 in alternative 0 is invalid",
        ):
            TurnData(
                turn_id="test_actual_empty_sequence",
                query="Test query",
                expected_tool_calls=[
                    [{"tool_name": "search", "arguments": {}}],  # Valid sequence
                    [],  # Empty sequence within single set - should be rejected
                ],
            )


class TestTurnDataFormatDetection:
    """Test cases for format detection logic."""

    def test_is_multiple_sets_format_empty(self):
        """Test format detection for empty lists."""
        # Empty list gets converted to [[]] which violates "only empty" constraint
        with pytest.raises(
            ValidationError, match="Empty set cannot be the only alternative"
        ):
            TurnData(turn_id="test", query="Test", expected_tool_calls=[])

    def test_is_single_set_format_multiple_empty(self):
        """Test format detection for multiple empty lists."""
        # [[], []] should be detected as multiple sets format (not single set)
        is_single = TurnData._is_single_set_format([[], []])
        assert is_single is False  # Should be treated as multiple sets format

    def test_is_single_set_format_detection(self):
        """Test detection of single set format."""
        # This should be detected as single set and converted
        turn_data = TurnData(
            turn_id="test",
            query="Test",
            expected_tool_calls=[
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ],
        )

        # Should be converted to multiple sets format with one set
        expected = turn_data.expected_tool_calls
        assert expected is not None
        assert len(expected) == 1  # One alternative set
        assert len(expected[0]) == 2  # Two sequences in that set
