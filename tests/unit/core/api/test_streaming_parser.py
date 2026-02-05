"""Unit tests for streaming parser."""

from typing import Any

import pytest

from lightspeed_evaluation.core.api.streaming_parser import (
    parse_streaming_response,
    _parse_sse_line,
    _parse_tool_call,
    _format_tool_sequences,
)


class TestParseStreamingResponse:
    """Unit tests for parse_streaming_response."""

    def test_parse_complete_response(self, mock_response: Any) -> None:
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
        # Performance metrics should be present
        assert "time_to_first_token" in result
        assert "streaming_duration" in result
        assert "tokens_per_second" in result

    def test_parse_response_with_tool_calls(self, mock_response: Any) -> None:
        """Test parsing response with tool calls."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_456"}}',
            'data: {"event": "tool_call", "data": '
            '{"token": {"tool_name": "search", "arguments": {"query": "test"}}}}',
            'data: {"event": "turn_complete", "data": {"token": "Final response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Final response"
        assert result["conversation_id"] == "conv_456"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0][0]["tool_name"] == "search"

    def test_parse_response_missing_final_response(self, mock_response: Any) -> None:
        """Test parsing fails when final response is missing."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_789"}}',
        ]
        mock_response.iter_lines.return_value = lines

        with pytest.raises(ValueError, match="No final response found"):
            parse_streaming_response(mock_response)

    def test_parse_response_missing_conversation_id(self, mock_response: Any) -> None:
        """Test parsing fails when conversation ID is missing."""
        lines = [
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        with pytest.raises(ValueError, match="No Conversation ID found"):
            parse_streaming_response(mock_response)

    def test_parse_response_with_error_event(self, mock_response: Any) -> None:
        """Test parsing handles error events."""
        lines = [
            'data: {"event": "error", "data": {"token": "API Error occurred"}}',
        ]
        mock_response.iter_lines.return_value = lines

        with pytest.raises(ValueError, match="Streaming API error: API Error occurred"):
            parse_streaming_response(mock_response)

    def test_parse_response_skips_empty_lines(self, mock_response: Any) -> None:
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

    def test_parse_response_skips_non_data_lines(self, mock_response: Any) -> None:
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

    def test_parse_response_with_multiple_tool_calls(self, mock_response: Any) -> None:
        """Test parsing multiple tool calls."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "tool_call", "data": '
            '{"token": {"tool_name": "search", "arguments": {"q": "test"}}}}',
            'data: {"event": "tool_call", "data": '
            '{"token": {"tool_name": "calculate", "arguments": {"expr": "2+2"}}}}',
            'data: {"event": "turn_complete", "data": {"token": "Done"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0][0]["tool_name"] == "search"
        assert result["tool_calls"][1][0]["tool_name"] == "calculate"

    def test_parse_response_with_new_format_tool_calls(
        self, mock_response: Any
    ) -> None:
        """Test parsing tool calls with new format (name/args directly in data)."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_new"}}',
            'data: {"event": "tool_call", "data": '
            '{"id": "tc_1", "name": "pods_list", "args": {"namespace": "default"}}}',
            'data: {"event": "tool_result", "data": '
            '{"id": "tc_1", "status": "success", "content": "pod/nginx Running"}}',
            'data: {"event": "turn_complete", "data": {"token": "Found pods"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert result["response"] == "Found pods"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0][0]["tool_name"] == "pods_list"
        assert result["tool_calls"][0][0]["arguments"]["namespace"] == "default"
        # Tool result should be associated with the tool call
        assert result["tool_calls"][0][0]["result"] == "pod/nginx Running"

    def test_parse_response_with_multiple_new_format_tool_calls(
        self, mock_response: Any
    ) -> None:
        """Test parsing multiple tool calls with new format and results."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_multi"}}',
            'data: {"event": "tool_call", "data": '
            '{"id": "tc_1", "name": "mcp_list_tools", "args": {"server_label": "kube"}}}',
            'data: {"event": "tool_result", "data": '
            '{"id": "tc_1", "status": "success", "content": "[tools list]"}}',
            'data: {"event": "tool_call", "data": {"id": "tc_2", '
            '"name": "pods_list_in_namespace", "args": {"namespace": "aladdin"}}}',
            'data: {"event": "tool_result", "data": '
            '{"id": "tc_2", "status": "success", "content": "pod list output"}}',
            'data: {"event": "turn_complete", "data": {"token": "Done"}}',
            'data: {"event": "end", "data": {"input_tokens": 100, "output_tokens": 50}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0][0]["tool_name"] == "mcp_list_tools"
        assert result["tool_calls"][0][0]["result"] == "[tools list]"
        assert result["tool_calls"][1][0]["tool_name"] == "pods_list_in_namespace"
        assert result["tool_calls"][1][0]["result"] == "pod list output"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50


class TestParseSSELine:
    """Unit tests for _parse_sse_line."""

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON SSE line."""
        json_data = '{"event": "start", "data": {"conversation_id": "123"}}'

        result = _parse_sse_line(json_data)

        assert result is not None
        event, data = result
        assert event == "start"
        assert data["conversation_id"] == "123"

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON returns None."""
        json_data = "not valid json"

        result = _parse_sse_line(json_data)

        assert result is None

    def test_parse_missing_event_field(self) -> None:
        """Test parsing with missing event field."""
        json_data = '{"data": {"some": "data"}}'

        result = _parse_sse_line(json_data)

        assert result is not None
        event, _ = result
        assert event == ""  # Default empty string

    def test_parse_missing_data_field(self) -> None:
        """Test parsing with missing data field."""
        json_data = '{"event": "test"}'

        result = _parse_sse_line(json_data)

        assert result is not None
        event, data = result
        assert event == "test"
        assert data == {}  # Default empty dict


class TestParseToolCall:
    """Unit tests for _parse_tool_call."""

    def test_parse_valid_tool_call(self) -> None:
        """Test parsing valid tool call with legacy format."""
        token = {"tool_name": "search", "arguments": {"query": "test"}}

        result = _parse_tool_call(token)

        assert result is not None
        assert result["tool_name"] == "search"
        assert result["arguments"]["query"] == "test"

    def test_parse_valid_tool_call_new_format(self) -> None:
        """Test parsing valid tool call with new name/args format."""
        token = {"name": "pods_list", "args": {"namespace": "default"}}

        result = _parse_tool_call(token)

        assert result is not None
        assert result["tool_name"] == "pods_list"
        assert result["arguments"]["namespace"] == "default"

    def test_parse_tool_call_missing_tool_name(self) -> None:
        """Test parsing tool call without tool_name."""
        token = {"arguments": {"query": "test"}}

        result = _parse_tool_call(token)

        assert result is None

    def test_parse_tool_call_missing_arguments(self) -> None:
        """Test parsing tool call without arguments defaults to empty dict."""
        token = {"tool_name": "search"}

        result = _parse_tool_call(token)

        # Missing arguments defaults to empty dict
        assert result is not None
        assert result["tool_name"] == "search"
        assert result["arguments"] == {}

    def test_parse_tool_call_with_empty_arguments(self) -> None:
        """Test parsing tool call with empty arguments dict."""
        token = {"tool_name": "search", "arguments": {}}

        result = _parse_tool_call(token)

        assert result is not None
        assert result["tool_name"] == "search"
        assert result["arguments"] == {}

    def test_parse_tool_call_invalid_structure(self) -> None:
        """Test parsing malformed tool call."""
        token: Any = "not a dict"

        result = _parse_tool_call(token)

        assert result is None


class TestFormatToolSequences:
    """Unit tests for _format_tool_sequences."""

    def test_format_empty_tool_calls(self) -> None:
        """Test formatting empty tool calls list."""
        result = _format_tool_sequences([])

        assert result == []

    def test_format_single_tool_call(self) -> None:
        """Test formatting single tool call."""
        tool_calls = [{"tool_name": "search", "arguments": {"query": "test"}}]

        result = _format_tool_sequences(tool_calls)

        assert len(result) == 1
        assert len(result[0]) == 1
        assert result[0][0]["tool_name"] == "search"

    def test_format_multiple_tool_calls(self) -> None:
        """Test formatting multiple tool calls into sequences."""
        tool_calls = [
            {"tool_name": "search", "arguments": {"query": "test"}},
            {"tool_name": "calculate", "arguments": {"expr": "2+2"}},
        ]

        result = _format_tool_sequences(tool_calls)

        assert len(result) == 2
        assert result[0][0]["tool_name"] == "search"
        assert result[1][0]["tool_name"] == "calculate"


class TestStreamingPerformanceMetrics:
    """Unit tests for streaming performance metrics (TTFT, tokens per second)."""

    def test_time_to_first_token_captured(self, mock_response: Any) -> None:
        """Test that time to first token is captured on first content event."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        # TTFT should be captured (non-None value)
        assert result["time_to_first_token"] is not None
        assert result["time_to_first_token"] >= 0

    def test_streaming_duration_captured(self, mock_response: Any) -> None:
        """Test that streaming duration is captured."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        # Streaming duration should be captured
        assert result["streaming_duration"] is not None
        assert result["streaming_duration"] >= 0
        # Duration should be >= TTFT
        assert result["streaming_duration"] >= result["time_to_first_token"]

    def test_tokens_per_second_with_token_counts(self, mock_response: Any) -> None:
        """Test tokens per second calculation when token counts are provided."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
            'data: {"event": "end", "data": {"input_tokens": 10, "output_tokens": 50}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        # Token counts should be captured
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 50
        # Tokens per second should be calculated (output_tokens > 0)
        assert result["tokens_per_second"] is not None
        assert result["tokens_per_second"] > 0

    def test_tokens_per_second_without_token_counts(self, mock_response: Any) -> None:
        """Test tokens per second is None when no output tokens."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "turn_complete", "data": {"token": "Response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        # Without output tokens, tokens_per_second should be None
        assert result["output_tokens"] == 0
        assert result["tokens_per_second"] is None

    def test_ttft_captured_on_token_event(self, mock_response: Any) -> None:
        """Test TTFT is captured on first token event (not just turn_complete)."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "token", "data": {"token": "partial"}}',
            'data: {"event": "turn_complete", "data": {"token": "Final response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        # TTFT should be captured on first content event (token)
        assert result["time_to_first_token"] is not None
        assert result["time_to_first_token"] >= 0

    def test_ttft_captured_on_tool_call_event(self, mock_response: Any) -> None:
        """Test TTFT is captured on tool_call event."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_123"}}',
            'data: {"event": "tool_call", "data": '
            '{"token": {"tool_name": "search", "arguments": {}}}}',
            'data: {"event": "turn_complete", "data": {"token": "Final response"}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        # TTFT should be captured on first content event (tool_call)
        assert result["time_to_first_token"] is not None
        assert result["time_to_first_token"] >= 0

    def test_performance_metrics_with_complete_flow(self, mock_response: Any) -> None:
        """Test complete streaming flow with all performance metrics."""
        lines = [
            'data: {"event": "start", "data": {"conversation_id": "conv_perf_test"}}',
            'data: {"event": "token", "data": {"token": "Streaming..."}}',
            'data: {"event": "tool_call", "data": '
            '{"token": {"tool_name": "search", "arguments": {"q": "test"}}}}',
            'data: {"event": "turn_complete", "data": {"token": "Complete response"}}',
            'data: {"event": "end", "data": {"input_tokens": 100, "output_tokens": 250}}',
        ]
        mock_response.iter_lines.return_value = lines

        result = parse_streaming_response(mock_response)

        # Verify all performance metrics are present
        assert result["response"] == "Complete response"
        assert result["conversation_id"] == "conv_perf_test"
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 250
        assert result["time_to_first_token"] is not None
        assert result["streaming_duration"] is not None
        assert result["tokens_per_second"] is not None
        # Verify tokens per second is reasonable (> 0)
        assert result["tokens_per_second"] > 0
