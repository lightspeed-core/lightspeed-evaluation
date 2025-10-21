"""Tests for streaming response parser utilities."""

from pytest_mock import MockerFixture

import pytest

from lsc_agent_eval.core.utils.streaming_parser import (
    _parse_tool_call,
    parse_streaming_response,
)


class TestStreamingResponseParser:
    """Test cases for streaming parser functions."""

    def test_basic_streaming_response(self, mocker: MockerFixture):
        """Test basic streaming response parsing."""
        mock_response = mocker.Mock()
        mock_response.iter_lines.return_value = [
            'data: {"event": "start", "data": {"conversation_id": "conv-123"}}',
            'data: {"event": "turn_complete", "data": {"token": "Hello world"}}',
        ]

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Hello world"
        assert result["conversation_id"] == "conv-123"
        assert result["tool_calls"] == []

    def test_streaming_with_tool_calls(self, mocker: MockerFixture):
        """Test streaming response with tool calls extraction."""
        mock_response = mocker.Mock()
        mock_response.iter_lines.return_value = [
            'data: {"event": "start", "data": {"conversation_id": "conv-tools"}}',
            'data: {"event": "tool_call", "data": {"token": {"tool_name": "create_pod", "arguments": {"name": "test"}}}}',
            'data: {"event": "turn_complete", "data": {"token": "Task completed"}}',
        ]

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Task completed"
        assert result["conversation_id"] == "conv-tools"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0][0]["tool_name"] == "create_pod"
        assert result["tool_calls"][0][0]["arguments"] == {"name": "test"}

    def test_error_conditions(self, mocker: MockerFixture):
        """Test error conditions - missing required data."""
        # Missing final response
        mock_response = mocker.Mock()
        mock_response.iter_lines.return_value = [
            'data: {"event": "start", "data": {"conversation_id": "conv-error"}}',
        ]

        with pytest.raises(ValueError, match="No final response found"):
            parse_streaming_response(mock_response)

        # Missing conversation ID
        mock_response.iter_lines.return_value = [
            'data: {"event": "turn_complete", "data": {"token": "Response without conv ID"}}',
        ]

        with pytest.raises(ValueError, match="No Conversation ID found"):
            parse_streaming_response(mock_response)

    def test_error_event_handling(self, mocker: MockerFixture):
        """Test error event handling in streaming response."""
        mock_response = mocker.Mock()
        mock_response.iter_lines.return_value = [
            'data: {"event": "start", "data": {"conversation_id": "conv-error"}}',
            'data: {"event": "error", "data": {"id": 1, "token": "Unable to connect to LLama Stack backend: Connection timed out"}}',
        ]

        with pytest.raises(
            ValueError,
            match="Streaming API error: Unable to connect to LLama Stack backend: Connection timed out",
        ):
            parse_streaming_response(mock_response)

    def test_tool_call_parsing(self):
        """Test tool call parsing functionality."""
        # Valid tool call
        result = _parse_tool_call({"tool_name": "list_versions", "arguments": {}})
        assert result["tool_name"] == "list_versions"
        assert result["arguments"] == {}

        # Tool call with arguments
        result = _parse_tool_call(
            {"tool_name": "create_pod", "arguments": {"name": "test"}}
        )
        assert result["tool_name"] == "create_pod"
        assert result["arguments"] == {"name": "test"}

        # Invalid format - missing tool_name
        assert _parse_tool_call({"arguments": {}}) is None
        # Invalid format - missing arguments field
        assert _parse_tool_call({"tool_name": "list_versions"}) is None

    def test_malformed_data_handling(self, mocker: MockerFixture):
        """Test handling of malformed data and edge cases."""
        mock_response = mocker.Mock()
        mock_response.iter_lines.return_value = [
            'data: {"event": "start", "data": {"conversation_id": "conv-malformed"}}',
            'data: {"invalid json": malformed}',  # Should be ignored
            "",  # Empty line - should be ignored
            "not a data line",  # Non-data line - should be ignored
            'data: {"event": "turn_complete", "data": {"token": "Success despite errors"}}',
        ]

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Success despite errors"
        assert result["conversation_id"] == "conv-malformed"
