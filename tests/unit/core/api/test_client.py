"""Unit tests for API client."""

import pytest

from lightspeed_evaluation.core.api.client import APIClient
from lightspeed_evaluation.core.models import APIConfig
from lightspeed_evaluation.core.system.exceptions import APIError


@pytest.fixture
def basic_api_config():
    """Create basic API configuration for streaming."""
    return APIConfig(
        enabled=True,
        api_base="http://localhost:8080",
        endpoint_type="streaming",
        timeout=30,
        provider="openai",
        model="gpt-4",
        cache_enabled=False,
    )


class TestAPIClient:
    """Unit tests for APIClient."""

    def test_initialization_streaming_endpoint(self, basic_api_config, mocker):
        """Test client initialization with streaming endpoint."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config)

        assert client.api_base == "http://localhost:8080"
        assert client.endpoint_type == "streaming"
        assert client.timeout == 30
        assert client.cache is None

    def test_initialization_with_cache(self, tmp_path, mocker):
        """Test client initialization with cache enabled."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080",
            endpoint_type="streaming",
            timeout=30,
            provider="openai",
            model="gpt-4",
            cache_enabled=True,
            cache_dir=str(tmp_path / "test_cache"),
        )

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")
        mock_cache = mocker.patch("lightspeed_evaluation.core.api.client.Cache")

        client = APIClient(config)

        assert client.cache is not None
        mock_cache.assert_called_once_with(str(tmp_path / "test_cache"))

    def test_validate_endpoint_type_valid(self, basic_api_config, mocker):
        """Test validation with valid endpoint type."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        # Should not raise error
        client = APIClient(basic_api_config)
        assert client.endpoint_type == "streaming"

    def test_setup_client_with_api_key(self, basic_api_config, mocker):
        """Test client setup includes API key from environment."""
        mocker.patch.dict("os.environ", {"API_KEY": "test_secret_key"})
        mock_client = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        APIClient(basic_api_config)

        # Verify headers were updated (should include Authorization header)
        assert mock_client.headers.update.call_count >= 1

    def test_query_requires_initialized_client(self, basic_api_config, mocker):
        """Test query fails if client not initialized."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config)
        client.client = None  # Simulate uninitialized client

        with pytest.raises(APIError, match="not initialized"):
            client.query("test query")

    def test_prepare_request_basic(self, basic_api_config, mocker):
        """Test request preparation with basic parameters."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config)
        request = client._prepare_request("What is Python?")

        assert request.query == "What is Python?"
        assert request.provider == "openai"
        assert request.model == "gpt-4"

    def test_prepare_request_with_conversation_id(self, basic_api_config, mocker):
        """Test request preparation with conversation ID."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config)
        request = client._prepare_request("Follow-up", conversation_id="conv_123")

        assert request.query == "Follow-up"
        assert request.conversation_id == "conv_123"

    def test_prepare_request_with_attachments(self, basic_api_config, mocker):
        """Test request preparation with attachments."""
        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(basic_api_config)
        request = client._prepare_request(
            "Analyze this", attachments=["file1.txt", "file2.pdf"]
        )

        assert request.query == "Analyze this"
        # Attachments may be processed, just verify they're present in some form
        assert hasattr(request, "attachments")

    def test_close_client(self, basic_api_config, mocker):
        """Test closing the HTTP client."""
        mock_http_client = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_http_client,
        )

        client = APIClient(basic_api_config)
        client.close()

        mock_http_client.close.assert_called_once()

    def test_get_cache_key_generates_consistent_hash(self, tmp_path, mocker):
        """Test cache key generation is consistent for same request."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080",
            endpoint_type="streaming",
            timeout=30,
            provider="openai",
            model="gpt-4",
            cache_enabled=True,
            cache_dir=str(tmp_path / "cache"),
        )

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")
        mocker.patch("lightspeed_evaluation.core.api.client.Cache")

        client = APIClient(config)

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
        self, basic_api_config, mocker
    ):
        """Test client initialization sets Content-Type header."""
        mock_client = mocker.Mock()
        mocker.patch(
            "lightspeed_evaluation.core.api.client.httpx.Client",
            return_value=mock_client,
        )

        APIClient(basic_api_config)

        # Verify Content-Type header was set
        calls = mock_client.headers.update.call_args_list
        assert any(
            "Content-Type" in str(call) or "application/json" in str(call)
            for call in calls
        )

    def test_standard_endpoint_initialization(self, mocker):
        """Test initialization with standard (non-streaming) endpoint."""
        config = APIConfig(
            enabled=True,
            api_base="http://localhost:8080",
            endpoint_type="query",
            timeout=30,
            provider="openai",
            model="gpt-4",
        )

        mocker.patch("lightspeed_evaluation.core.api.client.httpx.Client")

        client = APIClient(config)

        assert client.endpoint_type == "query"
