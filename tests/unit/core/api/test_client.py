# pylint: disable=protected-access

"""Unit tests for core API client module."""

from pathlib import Path
import pytest
import httpx
from pytest_mock import MockerFixture
from pydantic import ValidationError

from lightspeed_evaluation.core.models import APIConfig, APIResponse
from lightspeed_evaluation.core.system.exceptions import APIError
from lightspeed_evaluation.core.api.client import APIClient, _is_retryable_server_error


class TestAPIClient:
    """Unit tests for APIClient."""

    def test_initialization_unsupported_endpoint_type(self) -> None:
        """Test initialization fails with unsupported endpoint type."""

        # Pydantic will validate the endpoint_type, so this should raise ValidationError
        with pytest.raises(ValidationError, match="Endpoint type must be one of"):
            APIConfig(
                enabled=True,
                api_base="http://localhost:8080",
                version="v1",
                endpoint_type="unsupported_type",
                timeout=30,
            )

    def test_query_standard_endpoint_success(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
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

        client = APIClient(basic_api_config_query_endpoint)
        result = client.query("Test query")

        assert isinstance(result, APIResponse)
        assert result.response == "Test response"
        assert result.conversation_id == "conv_123"
        assert result.contexts == ["Context 1"]

    def test_query_with_conversation_id(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
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

        client = APIClient(basic_api_config_query_endpoint)
        result = client.query("Follow-up query", conversation_id="conv_123")

        assert result.conversation_id == "conv_123"

        # Check that conversation_id was passed in request
        call_kwargs = mock_client.post.call_args
        request_data = call_kwargs[1]["json"]
        assert request_data["conversation_id"] == "conv_123"

    def test_query_with_attachments(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
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

        client = APIClient(basic_api_config_query_endpoint)
        result = client.query("Query", attachments=["file1.txt", "file2.pdf"])

        assert result.response == "Response with attachments"

        # Check attachments in request - should be AttachmentData objects
        call_kwargs = mock_client.post.call_args
        request_data = call_kwargs[1]["json"]
        assert len(request_data["attachments"]) == 2
        assert request_data["attachments"][0]["content"] == "file1.txt"
        assert request_data["attachments"][1]["content"] == "file2.pdf"

    def test_query_http_error_non_retryable(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test query handling non-retryable HTTP errors (4xx except 429)."""
        mock_response = mocker.Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
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

        client = APIClient(basic_api_config_query_endpoint)

        with pytest.raises(APIError, match="API error: 400"):
            client.query("Test query")

    def test_query_timeout_error(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test query handling timeout."""

        mock_client = mocker.Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_query_endpoint)

        with pytest.raises(APIError, match="timeout"):
            client.query("Test query")

    def test_query_missing_response_field(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
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

        client = APIClient(basic_api_config_query_endpoint)

        with pytest.raises(APIError, match="missing 'response' field"):
            client.query("Test query")

    def test_query_streaming_endpoint(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test query to streaming endpoint."""
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

        client = APIClient(basic_api_config_streaming_endpoint)
        result = client.query("Test query")

        assert result.response == "Streamed response"
        assert result.conversation_id == "conv_123"

    def test_handle_response_errors_non_200(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test _handle_response_errors with non-200 status."""

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_query_endpoint)

        mock_response = mocker.Mock()
        mock_response.status_code = 404
        mock_response.request = mocker.Mock()
        mock_response.read.return_value = b'{"detail": "Not found"}'

        with pytest.raises(httpx.HTTPStatusError):
            client._handle_response_errors(mock_response)

    def test_extract_error_message_with_detail(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test _extract_error_message with detail field."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_query_endpoint)

        mock_response = mocker.Mock()
        mock_response.read.return_value = b'{"detail": "Error message"}'

        error_msg = client._extract_error_message(mock_response)
        assert "Error message" in error_msg

    def test_extract_error_message_with_nested_detail(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test _extract_error_message with nested detail."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_query_endpoint)

        mock_response = mocker.Mock()
        mock_response.read.return_value = (
            b'{"detail": {"response": "Error", "cause": "Reason"}}'
        )

        error_msg = client._extract_error_message(mock_response)
        assert "Error" in error_msg
        assert "Reason" in error_msg

    def test_standard_query_formats_tool_calls(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
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

        client = APIClient(basic_api_config_query_endpoint)
        result = client.query("Test query")

        # Tool calls should be formatted as list of lists
        assert len(result.tool_calls) == 2
        assert isinstance(result.tool_calls[0], list)
        assert result.tool_calls[0][0]["tool_name"] == "search"
        assert result.tool_calls[1][0]["tool_name"] == "calculator"


class TestAPIClientConfiguration:
    """Additional tests for APIClient configuration and initialization."""

    def test_initialization_streaming_endpoint(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test client initialization with streaming endpoint."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)

        assert client.config.api_base == "http://localhost:8080"
        assert client.config.endpoint_type == "streaming"
        assert client.config.timeout == 30
        assert client.cache is None

    def test_initialization_with_cache(
        self,
        basic_api_config_streaming_endpoint: APIConfig,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test client initialization with cache enabled."""
        # Enable cache in config and set cache directory
        basic_api_config_streaming_endpoint.cache_enabled = True
        basic_api_config_streaming_endpoint.cache_dir = str(tmp_path / "test_cache")

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")
        mock_cache = mocker.patch("lightspeed_evaluation.core.api.client.Cache")

        client = APIClient(basic_api_config_streaming_endpoint)

        assert client.cache is not None
        mock_cache.assert_called_once_with(str(tmp_path / "test_cache"))

    def test_validate_endpoint_type_valid(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test validation with valid endpoint type."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        # Should not raise error
        client = APIClient(basic_api_config_streaming_endpoint)
        assert client.config.endpoint_type == "streaming"

    def test_setup_client_with_api_key(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test client setup includes API key from environment."""
        mocker.patch.dict("os.environ", {"API_KEY": "test_secret_key"})
        mock_client = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        APIClient(basic_api_config_streaming_endpoint)

        # Verify headers were updated (should include Authorization header)
        assert mock_client.headers.update.call_count >= 1

    def test_query_requires_initialized_client(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test query fails if client not initialized."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)
        client.client = None  # Simulate uninitialized client

        with pytest.raises(APIError, match="not initialized"):
            client.query("test query")

    def test_prepare_request_basic(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test request preparation with basic parameters."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)
        request = client._prepare_request("What is Python?")

        assert request.query == "What is Python?"
        assert request.provider == "openai"
        assert request.model == "gpt-4"

    def test_prepare_request_with_conversation_id(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test request preparation with conversation ID."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)
        request = client._prepare_request("Follow-up", conversation_id="conv_123")

        assert request.query == "Follow-up"
        assert request.conversation_id == "conv_123"

    def test_prepare_request_with_attachments(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test request preparation with attachments."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)
        request = client._prepare_request(
            "Analyze this", attachments=["file1.txt", "file2.pdf"]
        )

        assert request.query == "Analyze this"
        # Attachments may be processed, just verify they're present in some form
        assert hasattr(request, "attachments")

    def test_close_client(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test closing the HTTP client."""
        mock_http_client = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_http_client,
        )

        client = APIClient(basic_api_config_streaming_endpoint)
        client.close()

        mock_http_client.close.assert_called_once()

    def test_get_cache_key_generates_consistent_hash(
        self,
        basic_api_config_streaming_endpoint: APIConfig,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        """Test cache key generation is consistent for same request."""
        # Enable cache in config and set cache directory
        basic_api_config_streaming_endpoint.cache_enabled = True
        basic_api_config_streaming_endpoint.cache_dir = str(tmp_path / "cache")

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")
        mocker.patch("lightspeed_evaluation.core.api.client.Cache")

        client = APIClient(basic_api_config_streaming_endpoint)

        # Create identical requests
        request1 = client._prepare_request("test query")
        request2 = client._prepare_request("test query")

        key1 = client._get_cache_key(request1)
        key2 = client._get_cache_key(request2)

        # Same request should generate same cache key
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) > 0

    def test_client_initialization_sets_content_type_header(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test client initialization sets Content-Type header."""
        mock_client = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        APIClient(basic_api_config_streaming_endpoint)

        # Verify Content-Type header was set
        calls = mock_client.headers.update.call_args_list
        assert any(
            "Content-Type" in str(call) or "application/json" in str(call)
            for call in calls
        )

    def test_standard_endpoint_initialization(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test initialization with standard (non-streaming) endpoint."""

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_query_endpoint)

        assert client.config.endpoint_type == "query"

    def test_get_cached_response_zeros_token_counts(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that _get_cached_response zeros out token counts."""
        basic_api_config_query_endpoint.cache_enabled = True

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        # Create a mock cache with a cached response that has token counts
        mock_cache = mocker.Mock()
        cached_response = APIResponse(
            response="Cached response",
            conversation_id="conv_123",
            input_tokens=50,
            output_tokens=100,
        )
        mock_cache.get.return_value = cached_response

        mocker.patch(
            "lightspeed_evaluation.core.api.client.Cache", return_value=mock_cache
        )

        client = APIClient(basic_api_config_query_endpoint)

        # Prepare a request
        request = client._prepare_request("Test query")

        # Get cached response
        result = client._get_cached_response(request)

        # Verify token counts were zeroed
        assert result is not None
        assert result.input_tokens == 0
        assert result.output_tokens == 0

        # Verify other fields remain unchanged
        assert result.response == "Cached response"
        assert result.conversation_id == "conv_123"


class TestExtraRequestParams:
    """Tests for extra_request_params support in APIClient."""

    def test_api_config_extra_request_params(self) -> None:
        """Test APIConfig accepts extra_request_params."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080",
            endpoint_type="query",
            timeout=30,
            extra_request_params={"mode": "troubleshooting"},
        )
        assert config.extra_request_params == {"mode": "troubleshooting"}

    def test_api_config_extra_request_params_none_default(self) -> None:
        """Test APIConfig defaults extra_request_params to None."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080",
            endpoint_type="query",
            timeout=30,
        )
        assert config.extra_request_params is None

    def test_prepare_request_with_turn_extra_params(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test request preparation with per-turn extra params."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)
        request = client._prepare_request(
            "Test query",
            extra_request_params={"mode": "troubleshooting"},
        )

        assert request.extra_request_params == {"mode": "troubleshooting"}

    def test_prepare_request_extra_params_falls_back_to_config(
        self, mocker: MockerFixture
    ) -> None:
        """Test extra params fall back to system config when not provided per-turn."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080",
            endpoint_type="query",
            timeout=30,
            cache_enabled=False,
            extra_request_params={"mode": "ask"},
        )
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(config)
        request = client._prepare_request("Test query")

        assert request.extra_request_params == {"mode": "ask"}

    def test_prepare_request_turn_extra_params_overrides_config(
        self, mocker: MockerFixture
    ) -> None:
        """Test per-turn extra params override system config values."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080",
            endpoint_type="query",
            timeout=30,
            cache_enabled=False,
            extra_request_params={"mode": "ask", "system_key": "default"},
        )
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(config)
        request = client._prepare_request(
            "Test query",
            extra_request_params={"mode": "troubleshooting"},
        )

        # Per-turn overrides mode, system_key is inherited
        assert request.extra_request_params == {
            "mode": "troubleshooting",
            "system_key": "default",
        }

    def test_cache_key_differs_by_extra_params(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that different extra params produce different cache keys."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)
        request_ask = client._prepare_request(
            "Test query", extra_request_params={"mode": "ask"}
        )
        request_troubleshooting = client._prepare_request(
            "Test query", extra_request_params={"mode": "troubleshooting"}
        )

        key_ask = client._get_cache_key(request_ask)
        key_troubleshooting = client._get_cache_key(request_troubleshooting)

        assert key_ask != key_troubleshooting

    def test_query_flattens_extra_params_in_payload(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that extra_request_params are flattened into the API payload."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Test response",
            "conversation_id": "conv_123",
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_query_endpoint)
        client.query("Test query", extra_request_params={"mode": "ask"})

        call_kwargs = mock_client.post.call_args
        request_data = call_kwargs[1]["json"]
        # mode should be flattened into top-level, not nested
        assert request_data["mode"] == "ask"
        assert "extra_request_params" not in request_data

    def test_serialize_request_skips_reserved_fields(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that extra_request_params cannot overwrite core request fields."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config_streaming_endpoint)
        request = client._prepare_request(
            "Original query",
            extra_request_params={"query": "injected", "mode": "troubleshooting"},
        )

        payload = client._serialize_request(request)

        # Core field must not be overwritten
        assert payload["query"] == "Original query"
        # Non-reserved field should be added
        assert payload["mode"] == "troubleshooting"


class TestRetryLogic:
    """Unit tests for retry logic in APIClient."""

    def test_is_retryable_server_error(self, mocker: MockerFixture) -> None:
        """Test _is_retryable_server_error identifies 429 and 5xx errors."""
        resp_429 = mocker.Mock(status_code=429)
        assert _is_retryable_server_error(
            httpx.HTTPStatusError("", request=mocker.Mock(), response=resp_429)
        )

        resp_500 = mocker.Mock(status_code=500)
        assert _is_retryable_server_error(
            httpx.HTTPStatusError("", request=mocker.Mock(), response=resp_500)
        )

        resp_502 = mocker.Mock(status_code=502)
        assert _is_retryable_server_error(
            httpx.HTTPStatusError("", request=mocker.Mock(), response=resp_502)
        )

        resp_503 = mocker.Mock(status_code=503)
        assert _is_retryable_server_error(
            httpx.HTTPStatusError("", request=mocker.Mock(), response=resp_503)
        )

        resp_400 = mocker.Mock(status_code=400)
        assert not _is_retryable_server_error(
            httpx.HTTPStatusError("", request=mocker.Mock(), response=resp_400)
        )

        resp_404 = mocker.Mock(status_code=404)
        assert not _is_retryable_server_error(
            httpx.HTTPStatusError("", request=mocker.Mock(), response=resp_404)
        )

        assert not _is_retryable_server_error(ValueError("not an HTTP error"))

    def test_standard_query_retries_on_429_then_succeeds(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test standard query retries on 429 error and then succeeds on retry."""
        mock_response_429 = mocker.Mock(status_code=429)
        mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 error", request=mocker.Mock(), response=mock_response_429
        )

        mock_response_success = mocker.Mock(status_code=200)
        mock_response_success.json.return_value = {
            "response": "Success after retry",
            "conversation_id": "conv_123",
        }

        mock_client = mocker.Mock()
        mock_client.post.side_effect = [mock_response_429, mock_response_success]
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_query_endpoint)
        result = client.query("Test standard query")

        assert result.response == "Success after retry"
        assert mock_client.post.call_count == 2

    def test_streaming_query_retries_on_429_then_succeeds(
        self, basic_api_config_streaming_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test streaming query retries on 429 error and then succeeds on retry."""
        mock_stream_429 = mocker.Mock(status_code=429, request=mocker.Mock())
        mock_context_429 = mocker.MagicMock()
        mock_context_429.__enter__.return_value = mock_stream_429

        mock_stream_success = mocker.Mock(status_code=200)
        mock_context_success = mocker.MagicMock()
        mock_context_success.__enter__.return_value = mock_stream_success

        mock_client = mocker.Mock(headers={})
        mock_client.stream.side_effect = [mock_context_429, mock_context_success]

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        mocker.patch(
            "lightspeed_evaluation.core.api.client.parse_streaming_response",
            return_value={
                "response": "Success after retry",
                "conversation_id": "conv_123",
            },
        )

        client = APIClient(basic_api_config_streaming_endpoint)
        result = client.query("Test streaming query")

        assert result.response == "Success after retry"
        assert mock_client.stream.call_count == 2

    def test_query_raises_api_error_after_max_retries(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test query raises APIError after exhausting retry attempts."""
        basic_api_config_query_endpoint.num_retries = 3

        mock_response_429 = mocker.Mock(status_code=429)
        mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 error", request=mocker.Mock(), response=mock_response_429
        )

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response_429
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_query_endpoint)

        with pytest.raises(APIError, match="Maximum retry attempts"):
            client.query("Test query")

        assert mock_client.post.call_count == 4  # 3 retries + 1 initial attempt

    def test_standard_query_retries_on_500_then_succeeds(
        self, basic_api_config_query_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test standard query retries on 500 error and succeeds on retry."""
        mock_response_500 = mocker.Mock(status_code=500, text="Internal server error")
        mock_response_500.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 error", request=mocker.Mock(), response=mock_response_500
        )

        mock_response_success = mocker.Mock(status_code=200)
        mock_response_success.json.return_value = {
            "response": "Success after 500 retry",
            "conversation_id": "conv_123",
        }

        mock_client = mocker.Mock()
        mock_client.post.side_effect = [mock_response_500, mock_response_success]
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_query_endpoint)
        result = client.query("Test standard query")

        assert result.response == "Success after 500 retry"
        assert mock_client.post.call_count == 2


class TestInferEndpoint:
    """Tests for RLSAPI /infer endpoint support."""

    def test_query_infer_endpoint_success(
        self, basic_api_config_infer_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test successful query to infer endpoint."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "text": "Infer response",
                "request_id": "req_abc",
                "input_tokens": 10,
                "output_tokens": 20,
            }
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_infer_endpoint)
        result = client.query("What is RHEL?")

        assert isinstance(result, APIResponse)
        assert result.response == "Infer response"
        assert result.conversation_id == "req_abc"
        assert result.input_tokens == 10
        assert result.output_tokens == 20

        call_kwargs = mock_client.post.call_args
        assert "/api/lightspeed/v1/infer" in call_kwargs[0][0]
        request_body = call_kwargs[1]["json"]
        assert request_body["question"] == "What is RHEL?"
        assert request_body["include_metadata"] is True

    def test_infer_query_formats_tool_calls(
        self, basic_api_config_infer_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that infer query formats tool calls correctly."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "text": "Response with tools",
                "request_id": "req_abc",
                "tool_calls": [
                    {
                        "id": "tc1",
                        "name": "search_documentation",
                        "args": {"q": "rhel"},
                    },
                    {"id": "tc2", "tool_name": "mcp_list_tools", "arguments": {}},
                ],
                "tool_results": [
                    {
                        "id": "tc1",
                        "type": "mcp_call",
                        "status": "success",
                        "content": "result1",
                    },
                    {
                        "id": "tc2",
                        "type": "tool_list",
                        "status": "completed",
                        "content": "tools",
                    },
                ],
            }
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_infer_endpoint)
        result = client.query("Test query")

        assert len(result.tool_calls) == 2
        assert isinstance(result.tool_calls[0], list)
        assert result.tool_calls[0][0]["tool_name"] == "search_documentation"
        assert result.tool_calls[0][0]["arguments"] == {"q": "rhel"}
        assert result.tool_calls[0][0]["result"] == "success"
        assert result.tool_calls[1][0]["tool_name"] == "mcp_list_tools"
        assert result.tool_calls[1][0]["result"] == "completed"

    def test_infer_query_extracts_rag_chunks(
        self, basic_api_config_infer_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test that infer query extracts RAG chunks from tool_results."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "text": "Response with RAG",
                "request_id": "req_abc",
                "tool_results": [
                    {
                        "id": "tr1",
                        "type": "mcp_call",
                        "status": "success",
                        "content": "Chunk one---Chunk two---Chunk three",
                    }
                ],
            }
        }

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_infer_endpoint)
        result = client.query("Test query")

        assert len(result.contexts) == 3
        assert "Chunk one" in result.contexts[0]
        assert "Chunk two" in result.contexts[1]
        assert "Chunk three" in result.contexts[2]

    def test_infer_query_missing_response_field(
        self, basic_api_config_infer_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test infer query handles missing response field."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {"request_id": "req_abc"}}

        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_infer_endpoint)

        with pytest.raises(APIError, match="missing 'response' field"):
            client.query("Test query")

    def test_infer_query_timeout_error(
        self, basic_api_config_infer_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test infer query handles timeout."""
        mock_client = mocker.Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_infer_endpoint)

        with pytest.raises(APIError, match="timeout"):
            client.query("Test query")

    def test_infer_query_retries_on_429(
        self, basic_api_config_infer_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test infer query retries on 429 then succeeds."""
        mock_response_429 = mocker.Mock(status_code=429)
        mock_response_429.raise_for_status.side_effect = httpx.HTTPStatusError(
            "429 error", request=mocker.Mock(), response=mock_response_429
        )

        mock_response_success = mocker.Mock(status_code=200)
        mock_response_success.json.return_value = {
            "data": {
                "text": "Success after retry",
                "request_id": "req_abc",
            }
        }

        mock_client = mocker.Mock()
        mock_client.post.side_effect = [mock_response_429, mock_response_success]
        mock_client.headers = {}

        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        client = APIClient(basic_api_config_infer_endpoint)
        result = client.query("Test query")

        assert result.response == "Success after retry"
        assert mock_client.post.call_count == 2

    def test_infer_query_http_error_non_retryable(
        self, basic_api_config_infer_endpoint: APIConfig, mocker: MockerFixture
    ) -> None:
        """Test infer query raises APIError for non-retryable HTTP errors."""
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

        client = APIClient(basic_api_config_infer_endpoint)

        with pytest.raises(APIError, match="API error: 400"):
            client.query("Test query")
