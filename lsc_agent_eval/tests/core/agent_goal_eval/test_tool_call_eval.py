"""Tests for tool call evaluation utilities."""

from unittest.mock import patch

from lsc_agent_eval.core.agent_goal_eval.tool_call_eval import compare_tool_calls


class TestToolCallEvaluator:
    """Test cases for tool call evaluation."""

    def test_simple_match(self):
        """Test simple tool call comparison."""
        expected = [[{"name": "list_versions", "arguments": {}}]]
        actual = [[{"name": "list_versions", "arguments": {}}]]

        assert compare_tool_calls(expected, actual)

    def test_with_arguments(self):
        """Test tool call comparison with arguments."""
        expected = [[{"name": "oc_get", "arguments": {"oc_get_args": ["namespaces"]}}]]
        actual = [[{"name": "oc_get", "arguments": {"oc_get_args": ["namespaces"]}}]]

        assert compare_tool_calls(expected, actual)

    def test_unordered_arguments(self):
        """Test unordered arguments comparison."""
        expected = [
            [
                {
                    "name": "create_pod",
                    "arguments": {"name": "test", "image": "nginx", "port": 80},
                }
            ]
        ]
        actual = [
            [
                {
                    "name": "create_pod",
                    "arguments": {"port": 80, "name": "test", "image": "nginx"},
                }
            ]
        ]

        assert compare_tool_calls(expected, actual)

    def test_multiple_sequences(self):
        """Test multiple tool call sequences."""
        expected = [
            [{"name": "list_versions", "arguments": {}}],
            [{"name": "create_pod", "arguments": {"name": "test"}}],
        ]
        actual = [
            [{"name": "list_versions", "arguments": {}}],
            [{"name": "create_pod", "arguments": {"name": "test"}}],
        ]

        assert compare_tool_calls(expected, actual)

    def test_wrong_tool_name(self):
        """Test wrong tool name fails."""
        expected = [[{"name": "list_versions", "arguments": {}}]]
        actual = [[{"name": "list_clusters", "arguments": {}}]]

        assert not compare_tool_calls(expected, actual)

    def test_wrong_argument_value(self):
        """Test wrong argument value fails."""
        expected = [
            [{"name": "create_pod", "arguments": {"name": "test", "image": "nginx"}}]
        ]
        actual = [
            [{"name": "create_pod", "arguments": {"name": "test", "image": "apache"}}]
        ]

        with patch(
            "lsc_agent_eval.core.agent_goal_eval.tool_call_eval.logger"
        ) as mock_logger:
            assert not compare_tool_calls(expected, actual)

            # Check that the specific argument mismatch was logged
            mock_logger.debug.assert_any_call(
                "Argument value mismatch for '%s': expected %s, got %s",
                "image",
                "nginx",
                "apache",
            )

    def test_missing_argument(self):
        """Test missing argument fails."""
        expected = [
            [{"name": "create_pod", "arguments": {"name": "test", "image": "nginx"}}]
        ]
        actual = [[{"name": "create_pod", "arguments": {"name": "test"}}]]

        assert not compare_tool_calls(expected, actual)

    def test_empty_sequences(self):
        """Test empty sequences match."""
        assert compare_tool_calls([], [])

    def test_missing_arguments_field(self):
        """Test missing arguments field is handled."""
        # This should never occur or API will give error for this
        # Update accordingly later
        expected = [[{"name": "simple_call", "arguments": {}}]]
        actual = [[{"name": "simple_call"}]]

        assert compare_tool_calls(expected, actual)
