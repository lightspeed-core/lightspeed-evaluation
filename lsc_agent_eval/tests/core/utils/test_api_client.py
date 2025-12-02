"""Tests for agent API client."""

import json
from pytest_mock import MockerFixture

import httpx
import pytest

from lsc_agent_eval.core.utils.api_client import AgentHttpClient
from lsc_agent_eval.core.utils.exceptions import AgentAPIError


class TestAgentHttpClient:
    """Test AgentHttpClient."""

    def test_init_without_token(self, mocker: MockerFixture):
        """Test initializing client without token."""
        mock_client = mocker.patch("httpx.Client")
        client = AgentHttpClient("http://localhost:8080/v1/")

        assert client.endpoint == "http://localhost:8080/v1/"
        mock_client.assert_called_once_with(
            base_url="http://localhost:8080/v1/", verify=False
        )

    def test_init_with_token_file(self, mocker: MockerFixture):
        """Test initializing client with token file."""
        token_content = "test-token-123"

        mock_client = mocker.patch("httpx.Client")
        mocker.patch("builtins.open", mocker.mock_open(read_data=token_content))

        client = AgentHttpClient("http://localhost:8080/v1/", "token.txt")

        assert client.endpoint == "http://localhost:8080/v1/"
        mock_client.assert_called_once_with(
            base_url="http://localhost:8080/v1/", verify=False
        )
        mock_client.return_value.headers.update.assert_called_once_with(
            {"Authorization": "Bearer test-token-123"}
        )

    def test_init_with_missing_token_file(self, mocker: MockerFixture):
        """Test initializing client with missing token file."""
        mocker.patch("httpx.Client")
        mocker.patch("builtins.open", side_effect=FileNotFoundError)

        with pytest.raises(AgentAPIError, match="Token file not found"):
            AgentHttpClient("http://localhost:8080/v1/", "missing.txt")

    def test_init_with_env_token(self, mocker: MockerFixture):
        """Test initializing client with environment token."""
        mock_client = mocker.patch("httpx.Client")
        mocker.patch("os.getenv", return_value="env-token-456")

        AgentHttpClient("http://localhost:8080/v1/")

        mock_client.return_value.headers.update.assert_called_once_with(
            {"Authorization": "Bearer env-token-456"}
        )

    def test_query_agent_success(self, mocker: MockerFixture):
        """Test successful agent query."""
        # Mock HTTP response
        mock_response = mocker.Mock()
        response_text = "There are 80 namespaces."
        tool_calls_data = [{"name": "oc_get", "args": {"oc_get_args": ["namespaces"]}}]
        mock_response.json.return_value = {
            "response": response_text,
            "conversation_id": "conv-id-123",
            "tool_calls": tool_calls_data,
        }
        mock_response.raise_for_status.return_value = None

        # Mock HTTP client
        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")

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
            [{"tool_name": "oc_get", "arguments": {"oc_get_args": ["namespaces"]}}]
        ]
        assert result["tool_calls"] == expected_formatted
        mock_client.post.assert_called_once_with(
            "/query",
            json=api_input,
            timeout=300,
        )

    def test_query_agent_success_empty_tool_calls(self, mocker: MockerFixture):
        """Test successful agent query with empty tool_calls."""
        # Mock HTTP response with empty tool_calls
        mock_response = mocker.Mock()
        response_text = "OpenShift Virtualization is an extension of the OpenShift Container Platform"
        mock_response.json.return_value = {
            "response": response_text,
            "conversation_id": "conv-id-123",
            "tool_calls": [],
        }
        mock_response.raise_for_status.return_value = None

        # Mock HTTP client
        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")

        api_input = {
            "query": "What is Openshift Virtualization?",
            "provider": "watsonx",
            "model": "ibm/granite-3-3-8b-instruct",
        }
        result = client.query_agent(api_input)

        assert result["response"] == response_text
        assert result["conversation_id"] == "conv-id-123"
        assert result["tool_calls"] == []

    def test_query_agent_http_error(self, mocker: MockerFixture):
        """Test agent query with HTTP error."""
        # Mock HTTP error response
        mock_response = mocker.Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        # Mock HTTP client
        mock_client = mocker.Mock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error", request=mocker.Mock(), response=mock_response
        )

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")

        api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}
        with pytest.raises(AgentAPIError, match="Agent API error: 500"):
            client.query_agent(api_input)

    def test_query_agent_timeout(self, mocker: MockerFixture):
        """Test agent query with timeout."""
        # Mock HTTP client
        mock_client = mocker.Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timeout")

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")

        api_input = {
            "query": "Test query",
            "provider": "agent_provider",
            "model": "agent_model",
        }
        with pytest.raises(AgentAPIError, match="Agent query timeout"):
            client.query_agent(api_input)

    def test_query_agent_missing_response_field(self, mocker: MockerFixture):
        """Test agent query with missing response field."""
        # Mock HTTP response without 'response' field
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"error": "Invalid request"}
        mock_response.raise_for_status.return_value = None

        # Mock HTTP client
        mock_client = mocker.Mock()
        mock_client.post.return_value = mock_response

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")

        api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}
        with pytest.raises(
            AgentAPIError, match="Agent response missing 'response' field"
        ):
            client.query_agent(api_input)

    def test_query_agent_client_not_initialized(self, mocker: MockerFixture):
        """Test agent query when client is not initialized."""
        mocker.patch("httpx.Client", side_effect=Exception("Setup failed"))
        with pytest.raises(AgentAPIError, match="Failed to setup HTTP client"):
            AgentHttpClient("http://localhost:8080/v1/")

    def test_close_client_success(self, mocker: MockerFixture):
        """Test closing client successfully."""
        mock_client = mocker.Mock()

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")
        client.close()

        mock_client.close.assert_called_once()

    def test_client_setup_exception(self, mocker: MockerFixture):
        """Test client setup exception."""
        mocker.patch("httpx.Client", side_effect=Exception("Setup failed"))
        with pytest.raises(
            AgentAPIError, match="Failed to setup HTTP client: Setup failed"
        ):
            AgentHttpClient("http://localhost:8080/v1/")

    # Streaming Query Tests
    def test_streaming_query_agent_success(self, mocker: MockerFixture):
        """Test successful streaming agent query."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        mock_stream_response = mocker.MagicMock()
        mock_stream_response.__enter__.return_value = mock_response
        mock_stream_response.__exit__.return_value = None

        mock_client = mocker.Mock()
        mock_client.stream.return_value = mock_stream_response

        # Expected parser result
        expected_result = {
            "response": "Hello World! This is the complete response.",
            "conversation_id": "stream-conv-123",
            "tool_calls": [],
        }

        mocker.patch("httpx.Client", return_value=mock_client)
        mock_parser = mocker.patch(
            "lsc_agent_eval.core.utils.api_client.parse_streaming_response"
        )

        mock_parser.return_value = expected_result
        client = AgentHttpClient("http://localhost:8080/v1/")

        api_input = {
            "query": "What is OpenShift?",
            "provider": "watsonx",
            "model": "ibm/granite-3-3-8b-instruct",
        }

        result = client.streaming_query_agent(api_input)
        assert result == expected_result

        mock_client.stream.assert_called_once_with(
            "POST",
            "/streaming_query",
            json=api_input,
            timeout=300,
        )

        mock_parser.assert_called_once_with(mock_response)

    def test_streaming_query_agent_parser_error(self, mocker: MockerFixture):
        """Test streaming agent query when parser raises ValueError."""
        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None

        mock_stream_response = mocker.MagicMock()
        mock_stream_response.__enter__.return_value = mock_response
        mock_stream_response.__exit__.return_value = None

        mock_client = mocker.Mock()
        mock_client.stream.return_value = mock_stream_response

        mocker.patch("httpx.Client", return_value=mock_client)
        mock_parser = mocker.patch(
            "lsc_agent_eval.core.utils.api_client.parse_streaming_response"
        )

        # Mock the parser to raise the specific error
        mock_parser.side_effect = ValueError("No Conversation ID found")

        client = AgentHttpClient("http://localhost:8080/v1/")
        api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}

        with pytest.raises(
            AgentAPIError,
            match="Streaming response validation error: No Conversation ID found",
        ):
            client.streaming_query_agent(api_input)

    def test_streaming_query_agent_timeout(self, mocker: MockerFixture):
        """Test streaming agent query with timeout."""
        # Mock HTTP client
        mock_client = mocker.Mock()
        mock_client.stream.side_effect = httpx.TimeoutException("Request timeout")

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")

        api_input = {
            "query": "Test query",
            "provider": "agent_provider",
            "model": "agent_model",
        }
        with pytest.raises(AgentAPIError, match="Agent streaming query timeout"):
            client.streaming_query_agent(api_input)

    def test_streaming_query_agent_http_error(self, mocker: MockerFixture):
        """Test streaming agent query with HTTP error."""
        error_response = {
            "detail": {
                "response": "Access denied",
                "cause": "You do not have permission to access this conversation",
            }
        }
        error_json = json.dumps(error_response)

        mock_response = mocker.Mock()
        mock_response.status_code = 403
        mock_response.read.return_value = error_json.encode("utf-8")

        mock_stream_response = mocker.MagicMock()
        mock_stream_response.__enter__.return_value = mock_response
        mock_stream_response.__exit__.return_value = None

        mock_client = mocker.Mock()
        mock_client.stream.return_value = mock_stream_response

        mocker.patch("httpx.Client", return_value=mock_client)
        client = AgentHttpClient("http://localhost:8080/v1/")

        api_input = {"query": "Test query", "provider": "openai", "model": "gpt-4"}

        with pytest.raises(
            AgentAPIError,
            match="Agent API error: 403 - Access denied - You do not have permission to access this conversation",
        ):
            client.streaming_query_agent(api_input)
