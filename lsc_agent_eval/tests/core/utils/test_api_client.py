"""Tests for agent API client."""

from unittest.mock import MagicMock, Mock, mock_open, patch

import httpx
import pytest

from lsc_agent_eval.core.utils.api_client import AgentHttpClient
from lsc_agent_eval.core.utils.exceptions import AgentAPIError


class TestAgentHttpClient:
    """Test AgentHttpClient."""

    def test_init_without_token(self):
        """Test initializing client without token."""
        with patch("httpx.Client") as mock_client:
            client = AgentHttpClient("http://localhost:8080")

            assert client.endpoint == "http://localhost:8080"
            mock_client.assert_called_once_with(
                base_url="http://localhost:8080", verify=False
            )

    def test_init_with_token_file(self):
        """Test initializing client with token file."""
        token_content = "test-token-123"

        with (
            patch("httpx.Client") as mock_client,
            patch("builtins.open", mock_open(read_data=token_content)),
        ):

            client = AgentHttpClient("http://localhost:8080", "token.txt")

            assert client.endpoint == "http://localhost:8080"
            mock_client.assert_called_once_with(
                base_url="http://localhost:8080", verify=False
            )
            mock_client.return_value.headers.update.assert_called_once_with(
                {"Authorization": "Bearer test-token-123"}
            )

    def test_init_with_missing_token_file(self):
        """Test initializing client with missing token file."""
        with (
            patch("httpx.Client"),
            patch("builtins.open", side_effect=FileNotFoundError),
        ):

            with pytest.raises(AgentAPIError, match="Token file not found"):
                AgentHttpClient("http://localhost:8080", "missing.txt")

    def test_init_with_env_token(self):
        """Test initializing client with environment token."""
        with (
            patch("httpx.Client") as mock_client,
            patch("os.getenv", return_value="env-token-456"),
        ):
            AgentHttpClient("http://localhost:8080")

            mock_client.return_value.headers.update.assert_called_once_with(
                {"Authorization": "Bearer env-token-456"}
            )

    def test_query_agent_success(self):
        """Test successful agent query."""
        # Mock HTTP response
        mock_response = Mock()
        response_text = "There are 80 namespaces."
        tool_calls_data = [{"name": "oc_get", "args": {"oc_get_args": ["namespaces"]}}]
        mock_response.json.return_value = {
            "response": response_text,
            "conversation_id": "conv-id-123",
            "tool_calls": tool_calls_data,
        }
        mock_response.raise_for_status.return_value = None

        # Mock HTTP client
        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            api_input = {
                "query": "How many namespaces are there?",
                "provider": "watsonx",
                "model": "ibm/granite-3-3-8b-instruct",
            }
            result = client.query_agent(api_input)

            assert result["response"] == response_text
            assert result["conversation_id"] == "conv-id-123"
            # Tool calls should be formatted into sequences
            expected_formatted = [
                [{"name": "oc_get", "arguments": {"oc_get_args": ["namespaces"]}}]
            ]
            assert result["tool_calls"] == expected_formatted
            mock_client.post.assert_called_once_with(
                "/v1/query",
                json=api_input,
                timeout=300,
            )

    def test_query_agent_success_empty_tool_calls(self):
        """Test successful agent query with empty tool_calls."""
        # Mock HTTP response with empty tool_calls
        mock_response = Mock()
        response_text = "OpenShift Virtualization is an extension of the OpenShift Container Platform"
        mock_response.json.return_value = {
            "response": response_text,
            "conversation_id": "conv-id-123",
            "tool_calls": [],
        }
        mock_response.raise_for_status.return_value = None

        # Mock HTTP client
        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            api_input = {
                "query": "What is Openshift Virtualization?",
                "provider": "watsonx",
                "model": "ibm/granite-3-3-8b-instruct",
            }
            result = client.query_agent(api_input)

            assert result["response"] == response_text
            assert result["conversation_id"] == "conv-id-123"
            assert result["tool_calls"] == []

    def test_query_agent_http_error(self):
        """Test agent query with HTTP error."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        # Mock HTTP client
        mock_client = Mock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=Mock(), response=mock_response
        )

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}
            with pytest.raises(AgentAPIError, match="Agent API error: 500"):
                client.query_agent(api_input)

    def test_query_agent_timeout(self):
        """Test agent query with timeout."""
        # Mock HTTP client
        mock_client = Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timeout")

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            api_input = {
                "query": "Test query",
                "provider": "agent_provider",
                "model": "agent_model",
            }
            with pytest.raises(AgentAPIError, match="Agent query timeout"):
                client.query_agent(api_input)

    def test_query_agent_missing_response_field(self):
        """Test agent query with missing response field."""
        # Mock HTTP response without 'response' field
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_response.raise_for_status.return_value = None

        # Mock HTTP client
        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}
            with pytest.raises(
                AgentAPIError, match="Agent response missing 'response' field"
            ):
                client.query_agent(api_input)

    def test_query_agent_client_not_initialized(self):
        """Test agent query when client is not initialized."""
        with patch("httpx.Client", side_effect=Exception("Setup failed")):
            with pytest.raises(AgentAPIError, match="Failed to setup HTTP client"):
                AgentHttpClient("http://localhost:8080")

    def test_close_client_success(self):
        """Test closing client successfully."""
        mock_client = Mock()

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")
            client.close()

            mock_client.close.assert_called_once()

    def test_client_setup_exception(self):
        """Test client setup exception."""
        with patch("httpx.Client", side_effect=Exception("Setup failed")):
            with pytest.raises(
                AgentAPIError, match="Failed to setup HTTP client: Setup failed"
            ):
                AgentHttpClient("http://localhost:8080")

    # Streaming Query Tests
    def test_streaming_query_agent_success(self):
        """Test successful streaming agent query."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        mock_stream_response = MagicMock()
        mock_stream_response.__enter__.return_value = mock_response
        mock_stream_response.__exit__.return_value = None

        mock_client = Mock()
        mock_client.stream.return_value = mock_stream_response

        # Expected parser result
        expected_result = {
            "response": "Hello World! This is the complete response.",
            "conversation_id": "stream-conv-123",
            "tool_calls": [],
        }

        with (
            patch("httpx.Client", return_value=mock_client),
            patch(
                "lsc_agent_eval.core.utils.api_client.parse_streaming_response"
            ) as mock_parser,
        ):

            mock_parser.return_value = expected_result
            client = AgentHttpClient("http://localhost:8080")

            api_input = {
                "query": "What is OpenShift?",
                "provider": "watsonx",
                "model": "ibm/granite-3-3-8b-instruct",
            }

            result = client.streaming_query_agent(api_input)
            assert result == expected_result

            mock_client.stream.assert_called_once_with(
                "POST",
                "/v1/streaming_query",
                json=api_input,
                timeout=300,
            )

            mock_parser.assert_called_once_with(mock_response)

    def test_streaming_query_agent_parser_error(self):
        """Test streaming agent query when parser raises ValueError."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None

        mock_stream_response = MagicMock()
        mock_stream_response.__enter__.return_value = mock_response
        mock_stream_response.__exit__.return_value = None

        mock_client = Mock()
        mock_client.stream.return_value = mock_stream_response

        with (
            patch("httpx.Client", return_value=mock_client),
            patch(
                "lsc_agent_eval.core.utils.api_client.parse_streaming_response"
            ) as mock_parser,
        ):

            # Mock the parser to raise the specific error
            mock_parser.side_effect = ValueError("No Conversation ID found")

            client = AgentHttpClient("http://localhost:8080")
            api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}

            with pytest.raises(
                AgentAPIError,
                match="Streaming response validation error: No Conversation ID found",
            ):
                client.streaming_query_agent(api_input)

    def test_streaming_query_agent_timeout(self):
        """Test streaming agent query with timeout."""
        # Mock HTTP client
        mock_client = Mock()
        mock_client.stream.side_effect = httpx.TimeoutException("Request timeout")

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            api_input = {
                "query": "Test query",
                "provider": "agent_provider",
                "model": "agent_model",
            }
            with pytest.raises(AgentAPIError, match="Agent streaming query timeout"):
                client.streaming_query_agent(api_input)

    def test_streaming_query_agent_http_error(self):
        """Test streaming agent query with HTTP error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = Mock()
        mock_client.stream.side_effect = httpx.HTTPStatusError(
            "Server error", request=Mock(), response=mock_response
        )

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}

            with pytest.raises(AgentAPIError, match="Agent API error: 500"):
                client.streaming_query_agent(api_input)
