"""Tests for agent API client."""

from unittest.mock import Mock, mock_open, patch

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
        mock_response.json.return_value = {"response": "Test agent response"}
        mock_response.raise_for_status.return_value = None

        # Mock HTTP client
        mock_client = Mock()
        mock_client.post.return_value = mock_response

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            result = client.query_agent("What is Kubernetes?", "openai", "gpt-4")

            assert result == "Test agent response"
            mock_client.post.assert_called_once_with(
                "/v1/query",
                json={
                    "query": "What is Kubernetes?",
                    "provider": "openai",
                    "model": "gpt-4",
                },
                timeout=300,
            )

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

            with pytest.raises(AgentAPIError, match="Agent API error: 500"):
                client.query_agent("Test query", "openai", "gpt-4")

    def test_query_agent_timeout(self):
        """Test agent query with timeout."""
        # Mock HTTP client
        mock_client = Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timeout")

        with patch("httpx.Client", return_value=mock_client):
            client = AgentHttpClient("http://localhost:8080")

            with pytest.raises(AgentAPIError, match="Agent query timeout"):
                client.query_agent("Test query", "openai", "gpt-4")

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

            with pytest.raises(
                AgentAPIError, match="Agent response missing 'response' field"
            ):
                client.query_agent("Test query", "openai", "gpt-4")

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
