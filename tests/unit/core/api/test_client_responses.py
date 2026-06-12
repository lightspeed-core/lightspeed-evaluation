# pylint: disable=protected-access,duplicate-code

"""Unit tests for APIClient /responses endpoint support."""

import httpx
import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.api.client import APIClient
from lightspeed_evaluation.core.models import APIConfig, APIResponse
from lightspeed_evaluation.core.system.exceptions import APIError


class TestResponsesEndpoint:
    """Tests for /responses endpoint support."""

    def test_query_responses_endpoint_success(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test successful query to responses endpoint with field mapping."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "Test response text.",
            "conversation": "conv-123",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        # Test that the client correctly maps response fields to APIResponse attributes
        client = APIClient(basic_api_config_responses_endpoint)
        result = client.query("test query")
        assert isinstance(result, APIResponse)
        assert result.response == "Test response text."
        assert result.conversation_id == "conv-123"
        assert result.input_tokens == 100
        assert result.output_tokens == 50

        # Verify the correct endpoint was called with preprocessed request body
        call_kwargs = mock_client.post.call_args
        request_body = call_kwargs[1]["json"]
        assert request_body["model"] == "openai/gpt-4o-mini"

    def test_responses_extra_request_params_merged(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that extra_request_params (e.g. tools) are merged into the request."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "Answer",
            "conversation": "conv_abc",
            "usage": {},
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        tools = [
            {"type": "file_search", "vector_store_ids": ["okp"]},
            {
                "type": "mcp",
                "server_label": "test-mcp",
                "server_url": "http://host.docker.internal:3000",
            },
        ]

        client = APIClient(basic_api_config_responses_endpoint)
        client.query(
            "test query",
            conversation_id="conv_abc",
            extra_request_params={"tools": tools, "store": True},
        )

        request_body = mock_client.post.call_args[1]["json"]
        assert request_body["tools"] == tools
        assert request_body["store"] is True

    def test_responses_extracts_file_search_call_as_tool_call_and_rag_chunk(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that file_search_call items produce both RAG chunks and tool_calls."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "Answer.",
            "conversation": "conv_abc",
            "usage": {},
            "output": [
                {"type": "mcp_list_tools", "tools": [{"name": "mock_tool"}]},
                {
                    "type": "file_search_call",
                    "queries": ["test search query"],
                    "results": [{"text": "Relevant chunk", "score": 1}],
                },
                {"type": "message", "content": []},
            ],
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)
        result = client.query("test query", conversation_id="conv_abc")

        assert result.contexts == ["Relevant chunk"]
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][0]["tool_name"] == "file_search"
        assert result.tool_calls[0][0]["arguments"] == {
            "queries": ["test search query"]
        }

    def test_responses_extracts_tool_calls_from_mcp_call_output_items(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that mcp_call output items are extracted as tool_calls."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "Answer.",
            "conversation": "conv_abc",
            "usage": {},
            "output": [
                {"type": "mcp_list_tools", "tools": [{"name": "mock_tool_e2e"}]},
                {
                    "type": "mcp_call",
                    "name": "mock_tool_e2e",
                    "arguments": {"message": "hello"},
                    "output": "tool result text",
                },
                {
                    "type": "file_search_call",
                    "results": [{"text": "RAG chunk", "score": 1}],
                },
            ],
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)
        result = client.query("test query", conversation_id="conv_abc")

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0][0]["tool_name"] == "mock_tool_e2e"
        assert result.tool_calls[0][0]["arguments"] == {"message": "hello"}
        assert result.tool_calls[0][0]["result"] == "tool result text"
        assert result.tool_calls[1][0]["tool_name"] == "file_search"
        assert result.contexts == ["RAG chunk"]

    def test_responses_mcp_call_with_error_field(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that mcp_call items with an error field capture it on the tool_call."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "Answer.",
            "conversation": "conv_abc",
            "usage": {},
            "output": [
                {
                    "type": "mcp_call",
                    "name": "jira_get_issue",
                    "arguments": {"issue_key": "PROJ-123"},
                    "error": "Tool execution failed: permission denied",
                },
            ],
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)
        result = client.query("test query", conversation_id="conv_abc")

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0][0]["tool_name"] == "jira_get_issue"
        assert (
            result.tool_calls[0][0]["error"]
            == "Tool execution failed: permission denied"
        )
        assert "result" not in result.tool_calls[0][0]

    def test_responses_missing_output_text_raises_error(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that missing output_text raises APIError."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "conversation": "conv_abc",
            "usage": {},
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)

        with pytest.raises(APIError, match="missing 'response' field"):
            client.query("test query", conversation_id="conv_abc")

    def test_responses_query_timeout_error(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test responses query handles timeout."""
        mock_client = mocker.Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)

        with pytest.raises(APIError, match="timeout"):
            client.query("test query", conversation_id="conv_abc")

    def test_responses_query_http_error_non_retryable(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test responses query raises APIError for non-retryable HTTP errors."""
        mock_response = mocker.Mock(status_code=400, text="Bad request")
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "400 error", request=mocker.Mock(), response=mock_response
        )

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)

        with pytest.raises(APIError, match="API error: 400"):
            client.query("test query", conversation_id="conv_abc")

    def test_responses_streaming_dispatches_to_parse_responses_streaming(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that stream=True in extra_request_params uses the streaming path."""
        mock_raw = {
            "response": "Streaming answer.",
            "conversation_id": "conv-stream-123",
            "input_tokens": 10,
            "output_tokens": 5,
            "tool_calls": [],
            "rag_chunks": [],
            "time_to_first_token": 0.1,
            "streaming_duration": 0.5,
            "tokens_per_second": 10.0,
        }
        mock_parse = mocker.patch(
            "lightspeed_evaluation.core.api.client.parse_responses_streaming",
            return_value=mock_raw,
        )

        mock_stream_response = mocker.MagicMock()
        mock_stream_response.status_code = 200
        mock_stream_cm = mocker.MagicMock()
        mock_stream_cm.__enter__ = mocker.Mock(return_value=mock_stream_response)
        mock_stream_cm.__exit__ = mocker.Mock(return_value=False)

        mock_client = mocker.Mock()
        mock_client.stream.return_value = mock_stream_cm
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)
        result = client.query("test query", extra_request_params={"stream": True})

        assert isinstance(result, APIResponse)
        assert result.response == "Streaming answer."
        assert result.conversation_id == "conv-stream-123"
        mock_client.stream.assert_called_once()
        mock_client.post.assert_not_called()
        mock_parse.assert_called_once_with(mock_stream_response)

    def test_responses_non_streaming_does_not_use_streaming_path(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that stream=False (default) uses the regular POST path."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output_text": "Regular answer.",
            "conversation": "conv-regular-456",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)
        result = client.query("test query")

        assert result.response == "Regular answer."
        mock_client.post.assert_called_once()
        mock_client.stream.assert_not_called()

    def test_responses_query_retries_on_429(
        self, basic_api_config_responses_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test responses query retries on 429 then succeeds."""
        mocker.patch("time.sleep")
        mock_response_429 = mocker.Mock(status_code=429)
        mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 error", request=mocker.Mock(), response=mock_response_429
        )

        mock_response_success = mocker.Mock(status_code=200)
        mock_response_success.json.return_value = {
            "output_text": "Success after retry",
            "conversation": "conv_abc",
            "usage": {},
        }

        mock_client = mocker.Mock()
        mock_client.post.side_effect = [mock_response_429, mock_response_success]
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_responses_endpoint)
        result = client.query("test query", conversation_id="conv_abc")

        assert result.response == "Success after retry"
        assert mock_client.post.call_count == 2
