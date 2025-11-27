"""Unit tests for streaming parser."""

import pytest

from lightspeed_evaluation.core.api.streaming_parser import (
    parse_streaming_response,
    _parse_sse_line,
    _parse_tool_call,
    _format_tool_sequences,
)


@pytest.fixture
def mock_response(mocker):
    """Create a mock streaming response."""
    response = mocker.Mock()
    return response


class TestParseStreamingResponse:
    """Unit tests for parse_streaming_response."""

    def test_parse_complete_response(self, mock_response):
        """Test parsing a complete streaming response."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "turn_complete", "data": {"token": "This is the response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert result["response"] == "This is the response"
        assert result["conversation_id"] == "conv_123"
        assert result["tool_calls"] == []

    def test_parse_response_with_tool_calls(self, mock_response):
        """Test parsing response with tool calls."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_456"}}',
            'data: {"event": "tool_call", "data": {"token": {"tool_name": "search", "arguments": {"query": "test"}}}}',
            'data: {"event": "turn_complete", "data": {"token": "Final response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Final response"
        assert result["conversation_id"] == "conv_456"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0][0]["tool_name"] == "search"

    def test_parse_response_missing_final_response(self, mock_response):
        """Test parsing fails when final response is missing."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_789"}}',
        ]
        mock_response.iter_lines.return_value = lines

        with pytest.raises(ValueError, match="No final response found"):
            parse_streaming_response(mock_response)

    def test_parse_response_missing_conversation_id(self, mock_response):
        """Test parsing fails when conversation ID is missing."""
        lines = [
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        with pytest.raises(ValueError, match="No Conversation ID found"):
            parse_streaming_response(mock_response)

    def test_parse_response_with_error_event(self, mock_response):
        """Test parsing handles error events."""
        lines = [
            'data: {"event": "error", "data": {"token": "API Error occurred"}}',
        ]
        mock_response.iter_lines.return_value = lines

        with pytest.raises(ValueError, match="Streaming API error: API Error occurred"):
            parse_streaming_response(mock_response)

    def test_parse_response_skips_empty_lines(self, mock_response):
        """Test parser skips empty lines."""
        lines = [
            "",
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            "",
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
            "",
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Response"
        assert result["conversation_id"] == "conv_123"

    def test_parse_response_skips_non_data_lines(self, mock_response):
        """Test parser skips lines without 'data:' prefix."""
        lines = [
            "event: start",
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            "event: turn_complete",
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Response"
        assert result["conversation_id"] == "conv_123"

    def test_parse_response_with_multiple_tool_calls(self, mock_response):
        """Test parsing multiple tool calls."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "tool_call", "data": {"token": {"tool_name": "search", "arguments": {"q": "test"}}}}',
            'data: {"event": "tool_call", "data": {"token": {"tool_name": "calculate", "arguments": {"expr": "2+2"}}}}',
            'data: {"event": "turn_complete", "data": {"token": "Done"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0][0]["tool_name"] == "search"
        assert result["tool_calls"][1][0]["tool_name"] == "calculate"


class TestParseSSELine:
    """Unit tests for _parse_sse_line."""

    def test_parse_valid_json(self):
        """Test parsing valid JSON SSE line."""
        json_data = '{"event": "start", "data": {"conversation_id": "123"}}'

        result = _parse_sse_line(json_data)

        assert result is not None
        event, data = result
        assert event == "start"
        assert data["conversation_id"] == "123"

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns None."""
        json_data = "not valid json"

        result = _parse_sse_line(json_data)

        assert result is None

    def test_parse_missing_event_field(self):
        """Test parsing with missing event field."""
        json_data = '{"data": {"some": "data"}}'

        result = _parse_sse_line(json_data)

        assert result is not None
        event, data = result
        assert event == ""  # Default empty string

    def test_parse_missing_data_field(self):
        """Test parsing with missing data field."""
        json_data = '{"event": "test"}'

        result = _parse_sse_line(json_data)

        assert result is not None
        event, data = result
        assert event == "test"
        assert data == {}  # Default empty dict


class TestParseToolCall:
    """Unit tests for _parse_tool_call."""

    def test_parse_valid_tool_call(self):
        """Test parsing valid tool call."""
        token = {"tool_name": "search", "arguments": {"query": "test"}}

        result = _parse_tool_call(token)

        assert result is not None
        assert result["tool_name"] == "search"
        assert result["arguments"]["query"] == "test"

    def test_parse_tool_call_missing_tool_name(self):
        """Test parsing tool call without tool_name."""
        token = {"arguments": {"query": "test"}}

        result = _parse_tool_call(token)

        assert result is None

    def test_parse_tool_call_missing_arguments(self):
        """Test parsing tool call without arguments."""
        token = {"tool_name": "search"}

        result = _parse_tool_call(token)

        assert result is None

    def test_parse_tool_call_with_empty_arguments(self):
        """Test parsing tool call with empty arguments dict."""
        token = {"tool_name": "search", "arguments": {}}

        result = _parse_tool_call(token)

        assert result is not None
        assert result["tool_name"] == "search"
        assert result["arguments"] == {}

    def test_parse_tool_call_invalid_structure(self):
        """Test parsing malformed tool call."""
        token = "not a dict"

        result = _parse_tool_call(token)

        assert result is None


class TestFormatToolSequences:
    """Unit tests for _format_tool_sequences."""

    def test_format_empty_tool_calls(self):
        """Test formatting empty tool calls list."""
        result = _format_tool_sequences([])

        assert result == []

    def test_format_single_tool_call(self):
        """Test formatting single tool call."""
        tool_calls = [{"tool_name": "search", "arguments": {"query": "test"}}]

        result = _format_tool_sequences(tool_calls)

        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0]["tool_name"] == "search"

    def test_format_multiple_tool_calls(self):
        """Test formatting multiple tool calls into sequences."""
        tool_calls = [
            {"tool_name": "search", "arguments": {"query": "test"}},
            {"tool_name": "calculate", "arguments": {"expr": "2+2"}},
        ]

        result = _format_tool_sequences(tool_calls)

        assert len(result) == 2
        assert result[0][0]["tool_name"] == "search"
        assert result[1][0]["tool_name"] == "calculate"
