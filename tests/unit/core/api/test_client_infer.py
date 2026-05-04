# pylint: disable=protected-access,duplicate-code

"""Unit tests for APIClient /infer endpoint support."""

import pytest
import httpx
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import APIConfig, APIResponse
from lightspeed_evaluation.core.system.exceptions import APIError
from lightspeed_evaluation.core.api.client import APIClient


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
                    {"id": "tc2", "name": "mcp_list_tools", "args": {}},
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
        assert result.tool_calls[0][0]["result"] == "result1"
        assert result.tool_calls[0][0]["status"] == "success"
        assert result.tool_calls[1][0]["tool_name"] == "mcp_list_tools"
        assert result.tool_calls[1][0]["result"] == "tools"
        assert result.tool_calls[1][0]["status"] == "completed"

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
