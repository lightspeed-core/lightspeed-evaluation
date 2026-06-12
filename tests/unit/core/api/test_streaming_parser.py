"""Unit tests for streaming parser."""

import json
from typing import Any

import pytest

from lightspeed_evaluation.core.api.streaming_parser import (
    _format_tool_sequences,
    _normalize_mcp_item,
    _parse_sse_line,
    _parse_tool_call,
    parse_responses_streaming,
    parse_streaming_response,
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


class TestNormalizeMcpItem:
    """Unit tests for _normalize_mcp_item."""

    def test_normalize_decodes_string_arguments(self) -> None:
        """Test JSON string arguments are decoded and output maps to result."""
        item = {
            "name": "get_issue",
            "arguments": '{"key": "VAL"}',
            "output": "data",
            "error": "denied",
        }

        result = _normalize_mcp_item(item)

        assert result == {
            "name": "get_issue",
            "arguments": {"key": "VAL"},
            "result": "data",
            "error": "denied",
        }

    def test_normalize_invalid_string_arguments_defaults_to_empty(self) -> None:
        """Test invalid JSON string arguments default to empty dict."""
        item = {"name": "get_issue", "arguments": "not json"}

        result = _normalize_mcp_item(item)

        assert result["arguments"] == {}


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


class TestParseResponsesStreaming:
    """Unit tests for parse_responses_streaming."""

    def _make_lines(self, events: list[dict]) -> list[str]:
        return [f"data: {json.dumps(e)}" for e in events]

    def test_parse_basic_response(self, mock_response: Any) -> None:
        """Test parsing a minimal responses streaming response."""
        mock_response.iter_lines.return_value = self._make_lines(
            [
                {"type": "response.created", "response": {"conversation": "conv-abc"}},
                {"type": "response.output_text.delta", "delta": "Hello "},
                {"type": "response.output_text.delta", "delta": "world"},
                {
                    "type": "response.completed",
                    "response": {
                        "output_text": "Hello world",
                        "usage": {"input_tokens": 10, "output_tokens": 2},
                        "conversation": "conv-abc",
                    },
                },
            ]
        )

        result = parse_responses_streaming(mock_response)

        assert result["response"] == "Hello world"
        assert result["conversation_id"] == "conv-abc"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 2
        assert not result["tool_calls"]
        assert not result["rag_chunks"]
        assert result["time_to_first_token"] is not None

    def test_parse_mcp_call_tool(self, mock_response: Any) -> None:
        """Test that mcp_call output items are extracted as tool_calls."""
        mock_response.iter_lines.return_value = self._make_lines(
            [
                {"type": "response.created", "response": {"conversation": "conv-1"}},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "mcp_call",
                        "name": "jira_get_issue",
                        "arguments": '{"issue_key": "PROJ-1"}',
                        "output": "issue data",
                    },
                },
                {"type": "response.output_text.delta", "delta": "Done"},
                {
                    "type": "response.completed",
                    "response": {
                        "output_text": "Done",
                        "usage": {"input_tokens": 5, "output_tokens": 1},
                    },
                },
            ]
        )

        result = parse_responses_streaming(mock_response)

        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0][0]
        assert tc["tool_name"] == "jira_get_issue"
        assert tc["arguments"] == {"issue_key": "PROJ-1"}
        assert tc["result"] == "issue data"

    def test_parse_mcp_call_with_error(self, mock_response: Any) -> None:
        """Test that mcp_call items with error field capture it."""
        mock_response.iter_lines.return_value = self._make_lines(
            [
                {"type": "response.created", "response": {"conversation": "conv-2"}},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "mcp_call",
                        "name": "jira_get_issue",
                        "arguments": "{}",
                        "error": "permission denied",
                    },
                },
                {"type": "response.output_text.delta", "delta": "Failed"},
                {
                    "type": "response.completed",
                    "response": {
                        "output_text": "Failed",
                        "usage": {"input_tokens": 5, "output_tokens": 1},
                    },
                },
            ]
        )

        result = parse_responses_streaming(mock_response)

        tc = result["tool_calls"][0][0]
        assert tc["error"] == "permission denied"
        assert "result" not in tc

    def test_parse_file_search_call(self, mock_response: Any) -> None:
        """Test that file_search_call items produce rag_chunks and a tool_call."""
        mock_response.iter_lines.return_value = self._make_lines(
            [
                {"type": "response.created", "response": {"conversation": "conv-3"}},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "file_search_call",
                        "queries": ["python async tutorial"],
                        "results": [{"text": "Some RAG chunk"}],
                    },
                },
                {"type": "response.output_text.delta", "delta": "Answer"},
                {
                    "type": "response.completed",
                    "response": {
                        "output_text": "Answer",
                        "usage": {"input_tokens": 8, "output_tokens": 1},
                    },
                },
            ]
        )

        result = parse_responses_streaming(mock_response)

        assert result["rag_chunks"] == [{"content": "Some RAG chunk"}]
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0][0]["tool_name"] == "file_search"
        assert result["tool_calls"][0][0]["arguments"] == {
            "queries": ["python async tutorial"]
        }

    def test_missing_response_raises(self, mock_response: Any) -> None:
        """Test that missing response text raises ValueError."""
        mock_response.iter_lines.return_value = self._make_lines(
            [
                {"type": "response.created", "response": {"conversation": "conv-4"}},
                {
                    "type": "response.completed",
                    "response": {
                        "output_text": "",
                        "usage": {},
                    },
                },
            ]
        )

        with pytest.raises(ValueError, match="No final response"):
            parse_responses_streaming(mock_response)

    def test_missing_conversation_id_raises(self, mock_response: Any) -> None:
        """Test that missing conversation_id raises ValueError."""
        mock_response.iter_lines.return_value = self._make_lines(
            [
                {"type": "response.output_text.delta", "delta": "hi"},
                {
                    "type": "response.completed",
                    "response": {
                        "output_text": "hi",
                        "usage": {},
                    },
                },
            ]
        )

        with pytest.raises(ValueError, match="No conversation_id"):
            parse_responses_streaming(mock_response)

    def test_done_sentinel_stops_parsing(self, mock_response: Any) -> None:
        """Test that [DONE] after response.completed stops parsing cleanly."""
        lines = self._make_lines(
            [
                {"type": "response.created", "response": {"conversation": "conv-5"}},
                {"type": "response.output_text.delta", "delta": "Hi"},
                {
                    "type": "response.completed",
                    "response": {
                        "output_text": "Hi",
                        "usage": {"input_tokens": 5, "output_tokens": 1},
                    },
                },
            ]
        ) + ["data: [DONE]"]
        mock_response.iter_lines.return_value = lines

        result = parse_responses_streaming(mock_response)

        assert result["response"] == "Hi"
        assert result["input_tokens"] == 5
        assert result["output_tokens"] == 1

    def test_done_sentinel_before_completed_raises(self, mock_response: Any) -> None:
        """Test that [DONE] before response.completed raises due to missing response."""
        created = {"type": "response.created", "response": {"conversation": "conv-6"}}
        delta = {"type": "response.output_text.delta", "delta": "Hi"}
        completed = {
            "type": "response.completed",
            "response": {
                "output_text": "should be ignored",
                "usage": {"input_tokens": 99, "output_tokens": 99},
            },
        }
        lines = [
            f"data: {json.dumps(created)}",
            f"data: {json.dumps(delta)}",
            "data: [DONE]",
            f"data: {json.dumps(completed)}",
        ]
        mock_response.iter_lines.return_value = lines

        with pytest.raises(ValueError, match="No final response"):
            parse_responses_streaming(mock_response)
