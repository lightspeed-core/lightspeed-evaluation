"""Tests for tool_eval module."""

from lightspeed_evaluation.core.metrics.custom.tool_eval import (
    evaluate_tool_calls,
    compare_tool_calls,
    _compare_tool_call_sequence,
    _compare_single_tool_call,
    _compare_tool_arguments,
)


class TestEvaluateToolCalls:
    """Test cases for evaluate_tool_calls function."""

    def test_primary_pattern_match(self) -> None:
        """Test successful match with primary pattern."""
        expected = [
            [  # Primary pattern
                [{"tool_name": "test_tool", "arguments": {"key": "value"}}]
            ]
        ]
        actual = [[{"tool_name": "test_tool", "arguments": {"key": "value"}}]]

        success, details = evaluate_tool_calls(expected, actual)

        assert success is True
        assert "Primary pattern matched" in details
        assert "Tool calls match expected structure and arguments" in details

    def test_alternative_pattern_match(self) -> None:
        """Test successful match with alternative pattern."""
        expected = [
            [  # Primary pattern
                [{"tool_name": "tool1", "arguments": {"key": "value1"}}]
            ],
            [  # Alternative pattern
                [{"tool_name": "tool2", "arguments": {"key": "value2"}}]
            ],
        ]
        actual = [[{"tool_name": "tool2", "arguments": {"key": "value2"}}]]

        success, details = evaluate_tool_calls(expected, actual)

        assert success is True
        assert "Alternative 2 matched" in details
        assert "Tool calls match expected structure and arguments" in details

    def test_empty_pattern_match_primary(self) -> None:
        """Test empty pattern match as primary."""
        expected: list[list[dict]] = [[]]  # Primary: no tools expected
        actual: list = []

        success, details = evaluate_tool_calls(
            expected, actual  # pyright: ignore[reportArgumentType]
        )

        assert success is True
        assert "Primary pattern matched" in details
        assert "No tool calls made (valid alternate skip scenario)" in details

    def test_empty_pattern_match_alternative(self) -> None:
        """Test empty pattern match as alternative."""
        expected = [
            [[{"tool_name": "test_tool", "arguments": {}}]],  # Primary: some tool
            [],  # Alternative: no tools (skip scenario)
        ]
        actual: list = []

        success, details = evaluate_tool_calls(expected, actual)

        assert success is True
        assert "Alternative 2 matched" in details
        assert "valid alternate skip scenario" in details

    def test_no_pattern_match(self) -> None:
        """Test when no patterns match."""
        expected = [
            [  # Primary pattern
                [{"tool_name": "tool1", "arguments": {"key": "value1"}}]
            ],
            [  # Alternative pattern
                [{"tool_name": "tool2", "arguments": {"key": "value2"}}]
            ],
        ]
        actual = [[{"tool_name": "tool3", "arguments": {"key": "value3"}}]]

        success, details = evaluate_tool_calls(expected, actual)

        assert success is False
        assert "didn't match any of the 2 expected pattern(s)" in details

    def test_error_handling(self) -> None:
        """Test error handling in evaluate_tool_calls."""
        # Invalid expected format should be handled gracefully
        expected = "invalid"  # Not a list
        actual: list = []

        success, details = evaluate_tool_calls(
            expected, actual  # pyright: ignore[reportArgumentType]
        )

        assert success is False
        # The function iterates over the string characters, so we get a different error
        assert (
            "not set as an expected alternative" in details
            or "Tool evaluation error" in details
        )


class TestCompareToolCalls:
    """Test cases for compare_tool_calls function."""

    def test_exact_match(self) -> None:
        """Test exact tool call match."""
        expected = [[{"tool_name": "test_tool", "arguments": {"key": "value"}}]]
        actual = [[{"tool_name": "test_tool", "arguments": {"key": "value"}}]]

        result = compare_tool_calls(expected, actual)

        assert result["success"] is True

    def test_length_mismatch(self) -> None:
        """Test tool call sequence length mismatch."""
        expected = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],
        ]
        actual = [[{"tool_name": "tool1", "arguments": {}}]]

        result = compare_tool_calls(expected, actual)

        assert result["success"] is False

    def test_empty_sequences(self) -> None:
        """Test empty tool call sequences."""
        expected: list = []
        actual: list = []

        result = compare_tool_calls(expected, actual)

        assert result["success"] is True


class TestCompareToolCallSequence:
    """Test cases for _compare_tool_call_sequence function."""

    def test_sequence_match(self) -> None:
        """Test matching tool call sequence."""
        expected = [
            {"tool_name": "tool1", "arguments": {"key1": "value1"}},
            {"tool_name": "tool2", "arguments": {"key2": "value2"}},
        ]
        actual = [
            {"tool_name": "tool1", "arguments": {"key1": "value1"}},
            {"tool_name": "tool2", "arguments": {"key2": "value2"}},
        ]

        result = _compare_tool_call_sequence(expected, actual)

        assert result is True

    def test_sequence_length_mismatch(self) -> None:
        """Test tool call sequence with different lengths."""
        expected = [{"tool_name": "tool1", "arguments": {}}]
        actual = [
            {"tool_name": "tool1", "arguments": {}},
            {"tool_name": "tool2", "arguments": {}},
        ]

        result = _compare_tool_call_sequence(expected, actual)

        assert result is False


class TestCompareSingleToolCall:
    """Test cases for _compare_single_tool_call function."""

    def test_tool_name_match(self) -> None:
        """Test matching tool names and arguments."""
        expected = {"tool_name": "test_tool", "arguments": {"key": "value"}}
        actual = {"tool_name": "test_tool", "arguments": {"key": "value"}}

        result = _compare_single_tool_call(expected, actual)

        assert result is True

    def test_tool_name_mismatch(self) -> None:
        """Test mismatched tool names."""
        expected = {"tool_name": "tool1", "arguments": {}}
        actual = {"tool_name": "tool2", "arguments": {}}

        result = _compare_single_tool_call(expected, actual)

        assert result is False

    def test_missing_arguments(self) -> None:
        """Test tool calls with missing arguments."""
        expected = {"tool_name": "test_tool", "arguments": {"key": "value"}}
        actual = {"tool_name": "test_tool"}  # Missing arguments

        result = _compare_single_tool_call(expected, actual)

        assert result is False  # Missing arguments cause mismatch


class TestCompareToolArguments:
    """Test cases for _compare_tool_arguments function."""

    def test_exact_arguments_match(self) -> None:
        """Test exact argument matching."""
        expected = {"key1": "value1", "key2": "value2"}
        actual = {"key1": "value1", "key2": "value2"}

        result = _compare_tool_arguments(expected, actual)

        assert result is True

    def test_regex_pattern_match(self) -> None:
        """Test regex pattern matching in arguments."""
        expected = {"name": "web-server-[0-9]+"}
        actual = {"name": "web-server-123"}

        result = _compare_tool_arguments(expected, actual)

        assert result is True

    def test_missing_argument_key(self) -> None:
        """Test missing argument key."""
        expected = {"key1": "value1", "key2": "value2"}
        actual = {"key1": "value1"}  # Missing key2

        result = _compare_tool_arguments(expected, actual)

        assert result is False

    def test_extra_argument_keys(self) -> None:
        """Test extra argument keys."""
        expected = {"key1": "value1"}
        actual = {"key1": "value1", "key2": "value2"}  # Extra key2

        result = _compare_tool_arguments(expected, actual)

        assert result is False

    def test_invalid_regex_pattern(self) -> None:
        """Test invalid regex pattern handling."""
        expected = {"name": "[invalid_regex"}  # Invalid regex
        actual = {"name": "test"}

        result = _compare_tool_arguments(expected, actual)

        assert result is False

    def test_non_dict_arguments(self) -> None:
        """Test non-dictionary arguments."""
        expected = "not_a_dict"
        actual = {"key": "value"}

        result = _compare_tool_arguments(
            expected, actual  # pyright: ignore[reportArgumentType]
        )

        assert result is False


class TestOrderedParameter:
    """Test cases for the ordered parameter in tool evaluation."""

    def test_ordered_true_default_matches_in_order(self) -> None:
        """Test ordered=True (default) matches when order is correct, fails otherwise."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual_correct = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],
        ]
        actual_wrong = [
            [{"tool_name": "tool2", "arguments": {}}],
            [{"tool_name": "tool1", "arguments": {}}],
        ]

        # Default (ordered=True) should match correct order
        success, details = evaluate_tool_calls(expected, actual_correct)
        assert success is True
        assert "ordered" in details

        # Should fail with wrong order
        success, _ = evaluate_tool_calls(expected, actual_wrong, ordered=True)
        assert success is False

    def test_ordered_false_matches_any_order(self) -> None:
        """Test ordered=False succeeds regardless of order."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual = [
            [{"tool_name": "tool2", "arguments": {}}],
            [{"tool_name": "tool1", "arguments": {}}],
        ]

        success, details = evaluate_tool_calls(expected, actual, ordered=False)
        assert success is True
        assert "unordered" in details

    def test_ordered_false_fails_when_content_differs(self) -> None:
        """Test ordered=False still fails when tool calls don't match."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual = [
            [{"tool_name": "tool3", "arguments": {}}],
            [{"tool_name": "tool4", "arguments": {}}],
        ]

        success, _ = evaluate_tool_calls(expected, actual, ordered=False)
        assert success is False

    def test_unordered_handles_duplicates_correctly(self) -> None:
        """Test unordered matching handles duplicate sequences properly."""
        # Each expected item must match exactly one actual item
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool1", "arguments": {}}],  # Duplicate
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual_valid = [
            [{"tool_name": "tool2", "arguments": {}}],
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool1", "arguments": {}}],
        ]
        actual_invalid = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],
            [{"tool_name": "tool3", "arguments": {}}],  # Missing second tool1
        ]

        assert evaluate_tool_calls(expected, actual_valid, ordered=False)[0] is True
        assert evaluate_tool_calls(expected, actual_invalid, ordered=False)[0] is False

    def test_tools_within_sequence_always_ordered(self) -> None:
        """Test that tools within a single sequence must always match in order.

        The `ordered` parameter only affects sequence order, not tool order within.
        """
        expected = [
            [  # Single sequence with two tools
                {"tool_name": "tool1", "arguments": {}},
                {"tool_name": "tool2", "arguments": {}},
            ]
        ]
        actual = [
            [  # Tools in wrong order within the sequence
                {"tool_name": "tool2", "arguments": {}},
                {"tool_name": "tool1", "arguments": {}},
            ]
        ]

        # Should fail because within a sequence, order always matters
        result = _compare_tool_call_sequence(expected[0], actual[0])
        assert result is False


class TestMatchParameter:
    """Test cases for full_match parameter (full vs partial matching)."""

    def test_full_match_default_requires_exact_count(self) -> None:
        """Test full_match=True (default) requires all expected to match all actual."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual_exact = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],
        ]
        actual_extra = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],
            [{"tool_name": "tool3", "arguments": {}}],  # Extra tool
        ]

        # Full match succeeds with exact count
        success, details = evaluate_tool_calls(expected, actual_exact)
        assert success is True
        assert "full" in details

        # Full match fails with extra tools
        success, _ = evaluate_tool_calls(expected, actual_extra, full_match=True)
        assert success is False

    def test_partial_match_allows_extra_actual_tools(self) -> None:
        """Test full_match=False allows extra actual tools."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
            ]
        ]
        actual = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],  # Extra - should be ignored
            [{"tool_name": "tool3", "arguments": {}}],  # Extra - should be ignored
        ]

        success, details = evaluate_tool_calls(expected, actual, full_match=False)
        assert success is True
        assert "partial" in details

    def test_partial_match_succeeds_with_some_matches(self) -> None:
        """Test full_match=False succeeds if any expected tool is found."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool3", "arguments": {}}],  # tool2 not found
        ]

        success, details = evaluate_tool_calls(expected, actual, full_match=False)
        # Should succeed because tool1 matched (1 out of 2)
        assert success is True
        assert "1/2 matched" in details
        assert "1 unmatched" in details

    def test_partial_match_fails_when_no_matches(self) -> None:
        """Test full_match=False fails when no expected tools are found."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual = [
            [{"tool_name": "tool3", "arguments": {}}],
            [{"tool_name": "tool4", "arguments": {}}],  # Neither tool1 nor tool2 found
        ]

        success, _ = evaluate_tool_calls(expected, actual, full_match=False)
        assert success is False

    def test_partial_match_ordered_reports_statistics(self) -> None:
        """Test full_match=False with ordered=True reports match statistics."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool3", "arguments": {}}],
            ]
        ]
        # tool1 before tool3, with tool2 in between - both should match
        actual = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],  # Extra in between
            [{"tool_name": "tool3", "arguments": {}}],
        ]

        success, details = evaluate_tool_calls(
            expected, actual, full_match=False, ordered=True
        )
        assert success is True
        assert "2/2 matched" in details
        assert "0 unmatched" in details

    def test_partial_match_ordered_finds_all_items(self) -> None:
        """Test full_match=False ordered finds all items using greedy matching."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool3", "arguments": {}}],
            ]
        ]
        # tool3 before tool1 - greedy matching finds both
        actual = [
            [{"tool_name": "tool3", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],
            [{"tool_name": "tool1", "arguments": {}}],
        ]

        success, details = evaluate_tool_calls(
            expected, actual, full_match=False, ordered=True
        )
        # Greedy matching finds tool1 at index 2, tool3 at index 0
        # Both expected items are found, regardless of positions
        assert success is True
        assert "2/2 matched" in details

    def test_partial_match_unordered_ignores_order(self) -> None:
        """Test full_match=False with ordered=False ignores order."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool3", "arguments": {}}],
            ]
        ]
        # tool3 before tool1 - both should match when unordered
        actual = [
            [{"tool_name": "tool3", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],  # Extra
            [{"tool_name": "tool1", "arguments": {}}],
        ]

        success, details = evaluate_tool_calls(
            expected, actual, full_match=False, ordered=False
        )
        assert success is True
        assert "partial" in details
        assert "unordered" in details
        assert "2/2 matched" in details

    def test_partial_match_all_matched_reports_correctly(self) -> None:
        """Test full_match=False reports all matched correctly."""
        expected = [
            [
                [{"tool_name": "tool1", "arguments": {}}],
                [{"tool_name": "tool2", "arguments": {}}],
            ]
        ]
        actual = [
            [{"tool_name": "tool1", "arguments": {}}],
            [{"tool_name": "tool2", "arguments": {}}],
            [{"tool_name": "tool3", "arguments": {}}],  # Extra
        ]

        success, details = evaluate_tool_calls(expected, actual, full_match=False)
        assert success is True
        assert "2/2 matched" in details
        assert "0 unmatched" in details
