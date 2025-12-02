"""Unit tests for core API client module."""

import pytest

from lightspeed_evaluation.core.models import APIConfig, APIResponse
from lightspeed_evaluation.core.system.exceptions import APIError
from lightspeed_evaluation.core.api.client import APIClient


@pytest.fixture
def api_config():
    """Create test API config."""
    return APIConfig(
        enabled=True,
        api_base="http://localhost:8080/v1/",
        endpoint_type="query",
        timeout=30,
        cache_enabled=False,
    )


class TestAPIClient:
    """Unit tests for APIClient."""

    def test_initialization_unsupported_endpoint_type(self):
        """Test initialization fails with unsupported endpoint type."""
        from pydantic import ValidationError

        # Pydantic will validate the endpoint_type, so this should raise ValidationError
        with pytest.raises(ValidationError, match="Endpoint type must be one of"):
            APIConfig(
                enabled=True,
                api_base="http://localhost:8080/v1/",
                endpoint_type="unsupported_type",
                timeout=30,
            )

    def test_query_standard_endpoint_success(self, api_config, mocker):
        """Test successful query to standard endpoint."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response",
            "conversation_id": "conv_123",
            "rag_chunks": [{"content": "Context 1"}],
            "tool_calls": [],
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(api_config)
        result = client.query("Test query")

        assert isinstance(result, APIResponse)
        assert result.response == "Test response"
        assert result.conversation_id == "conv_123"
        assert result.contexts == ["Context 1"]

    def test_query_with_conversation_id(self, api_config, mocker):
        """Test query with existing conversation_id."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Continued response",
            "conversation_id": "conv_123",
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(api_config)
        result = client.query("Follow-up query", conversation_id="conv_123")

        assert result.conversation_id == "conv_123"

        # Check that conversation_id was passed in request
        call_kwargs = mock_client.post.call_args
        request_data = call_kwargs[1]["json"]
        assert request_data["conversation_id"] == "conv_123"

    def test_query_with_attachments(self, api_config, mocker):
        """Test query with attachments."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Response with attachments",
            "conversation_id": "conv_123",
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(api_config)
        result = client.query("Query", attachments=["file1.txt", "file2.pdf"])

        assert result.response == "Response with attachments"

        # Check attachments in request - should be AttachmentData objects
        call_kwargs = mock_client.post.call_args
        request_data = call_kwargs[1]["json"]
        assert len(request_data["attachments"]) == 2
        assert request_data["attachments"][0]["content"] == "file1.txt"
        assert request_data["attachments"][1]["content"] == "file2.pdf"

    def test_query_http_error(self, api_config, mocker):
        """Test query handling HTTP errors."""
        import httpx

        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 error", request=mocker.Mock(), response=mock_response
        )

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(api_config)

        with pytest.raises(APIError, match="API error: 500"):
            client.query("Test query")

    def test_query_timeout_error(self, api_config, mocker):
        """Test query handling timeout."""
        import httpx

        mock_client = mocker.Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(api_config)

        with pytest.raises(APIError, match="timeout"):
            client.query("Test query")

    def test_query_missing_response_field(self, api_config, mocker):
        """Test query handling missing response field."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "conversation_id": "conv_123"
            # Missing 'response' field
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(api_config)

        with pytest.raises(APIError, match="missing 'response' field"):
            client.query("Test query")

    def test_query_streaming_endpoint(self, mocker):
        """Test query to streaming endpoint."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080/v1/",
            endpoint_type="streaming",
            timeout=30,
            cache_enabled=False,
        )

        # Mock streaming response
        mock_stream_response = mocker.Mock()
        mock_stream_response.status_code = 200

        # Mock the context manager
        mock_stream_context = mocker.MagicMock()
        mock_stream_context.__enter__.return_value = mock_stream_response
        mock_stream_context.__exit__.return_value = None

        mock_client = mocker.Mock()
        mock_client.stream.return_value = mock_stream_context
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        # Mock the streaming parser
        mock_parser = mocker.patch(
            "lightspeed_evaluation.core.api.client.parse_streaming_response"
        )
        mock_parser.return_value = {
            "response": "Streamed response",
            "conversation_id": "conv_123",
        }

        client = APIClient(config)
        result = client.query("Test query")

        assert result.response == "Streamed response"
        assert result.conversation_id == "conv_123"

    def test_handle_response_errors_non_200(self, api_config, mocker):
        """Test _handle_response_errors with non-200 status."""
        import httpx

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(api_config)

        mock_response = mocker.Mock()
        mock_response.status_code = 404
        mock_response.request = mocker.Mock()
        mock_response.read.return_value = b'{"detail": "Not found"}'

        with pytest.raises(httpx.HTTPStatusError):
            client._handle_response_errors(mock_response)

    def test_extract_error_message_with_detail(self, api_config, mocker):
        """Test _extract_error_message with detail field."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(api_config)

        mock_response = mocker.Mock()
        mock_response.read.return_value = b'{"detail": "Error message"}'

        error_msg = client._extract_error_message(mock_response)
        assert "Error message" in error_msg

    def test_extract_error_message_with_nested_detail(self, api_config, mocker):
        """Test _extract_error_message with nested detail."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(api_config)

        mock_response = mocker.Mock()
        mock_response.read.return_value = (
            b'{"detail": {"response": "Error", "cause": "Reason"}}'
        )

        error_msg = client._extract_error_message(mock_response)
        assert "Error" in error_msg
        assert "Reason" in error_msg

    def test_standard_query_formats_tool_calls(self, api_config, mocker):
        """Test that standard query formats tool calls correctly."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Response with tools",
            "conversation_id": "conv_123",
            "tool_calls": [
                {"tool_name": "search", "arguments": {"q": "test"}},
                {"name": "calculator", "args": {"expr": "2+2"}},
            ],
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(api_config)
        result = client.query("Test query")

        # Tool calls should be formatted as list of lists
        assert len(result.tool_calls) == 2
        assert isinstance(result.tool_calls[0], list)
        assert result.tool_calls[0][0]["tool_name"] == "search"
        assert result.tool_calls[1][0]["tool_name"] == "calculator"
