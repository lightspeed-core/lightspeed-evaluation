"""Additional tests for API models to increase coverage."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.api import (
    RAGChunk,
    AttachmentData,
    APIRequest,
    APIResponse,
)


class TestRAGChunk:
    """Tests for RAGChunk model."""

    def test_rag_chunk_creation(self) -> None:
        """Test creating RAG chunk."""
        chunk = RAGChunk(content="test content", source="test source", score=0.95)

        assert chunk.content == "test content"
        assert chunk.source == "test source"
        assert chunk.score == 0.95

    def test_rag_chunk_without_score(self) -> None:
        """Test RAG chunk without score."""
        chunk = RAGChunk(content="content", source="source")

        assert chunk.score is None

    def test_rag_chunk_extra_field_forbidden(self) -> None:
        """Test that extra fields are forbidden."""
        with pytest.raises(ValidationError):
            RAGChunk(
                content="content",
                source="source",
                extra_field="not allowed",  # pyright: ignore[reportCallIssue]
            )


class TestAttachmentData:
    """Tests for AttachmentData model."""

    def test_attachment_creation(self) -> None:
        """Test creating attachment."""
        attachment = AttachmentData(content="file content")

        assert attachment.content == "file content"
        assert attachment.attachment_type == "configuration"
        assert attachment.content_type == "text/plain"

    def test_attachment_custom_type(self) -> None:
        """Test attachment with custom types."""
        attachment = AttachmentData(
            content="yaml data",
            attachment_type="yaml",
            content_type="application/yaml",
        )

        assert attachment.attachment_type == "yaml"
        assert attachment.content_type == "application/yaml"


class TestAPIRequest:
    """Tests for APIRequest model."""

    def test_create_simple_request(self) -> None:
        """Test creating simple API request."""
        request = APIRequest.create(query="What is Python?")

        assert request.query == "What is Python?"
        assert request.provider is None
        assert request.model is None

    def test_create_request_with_all_params(self) -> None:
        """Test creating request with all parameters."""
        request = APIRequest.create(
            query="Test query",
            provider="openai",
            model="gpt-4",
            no_tools=True,
            conversation_id="conv123",
            system_prompt="Custom prompt",
        )

        assert request.query == "Test query"
        assert request.provider == "openai"
        assert request.model == "gpt-4"
        assert request.no_tools is True
        assert request.conversation_id == "conv123"
        assert request.system_prompt == "Custom prompt"

    def test_create_request_with_attachments(self) -> None:
        """Test creating request with attachments."""
        # APIRequest.create expects string attachments, not AttachmentData objects
        attachments = ["file1", "file2"]

        request = APIRequest.create(
            query="Test",
            attachments=attachments,
        )

        assert request.attachments is not None
        assert len(request.attachments) == 2
        assert (
            request.attachments[0].content  # pylint: disable=unsubscriptable-object
            == "file1"
        )

    def test_request_empty_query_validation(self) -> None:
        """Test that empty query fails validation."""
        with pytest.raises(ValidationError):
            APIRequest(query="")


class TestAPIResponse:
    """Tests for APIResponse model."""

    def test_response_creation(self) -> None:
        """Test creating API response."""
        response = APIResponse(
            response="Test response",
            conversation_id="conv123",
            contexts=["context1", "context2"],
        )

        assert response.response == "Test response"
        assert response.conversation_id == "conv123"
        assert len(response.contexts) == 2

    def test_response_empty_contexts(self) -> None:
        """Test response with empty contexts."""
        response = APIResponse(
            response="Test",
            conversation_id="conv123",
        )

        assert not response.contexts

    def test_response_with_tool_calls(self) -> None:
        """Test response with tool calls."""
        response = APIResponse(
            response="Test",
            conversation_id="conv123",
            tool_calls=[[{"name": "search", "args": {}}]],
        )

        assert len(response.tool_calls) == 1

    def test_from_raw_response(self) -> None:
        """Test creating response from raw API data."""
        raw_data = {
            "response": "Test response",
            "conversation_id": "conv123",
            "rag_chunks": [
                {"content": "chunk1", "source": "source1"},
                {"content": "chunk2", "source": "source2"},
            ],
        }

        response = APIResponse.from_raw_response(raw_data)

        assert response.response == "Test response"
        assert response.conversation_id == "conv123"
        assert len(response.contexts) == 2
        assert "chunk1" in response.contexts

    def test_from_raw_response_without_conversation_id(self) -> None:
        """Test that from_raw_response fails without conversation_id."""
        raw_data = {"response": "Test"}

        with pytest.raises(ValueError, match="conversation_id is required"):
            APIResponse.from_raw_response(raw_data)

    def test_response_with_streaming_performance_metrics(self) -> None:
        """Test response with streaming performance metrics."""
        response = APIResponse(
            response="Test",
            conversation_id="conv123",
            time_to_first_token=0.125,
            streaming_duration=2.5,
            tokens_per_second=85.3,
        )

        assert response.time_to_first_token == 0.125
        assert response.streaming_duration == 2.5
        assert response.tokens_per_second == 85.3

    def test_response_without_streaming_metrics(self) -> None:
        """Test response defaults for streaming metrics (None for non-streaming)."""
        response = APIResponse(
            response="Test",
            conversation_id="conv123",
        )

        assert response.time_to_first_token is None
        assert response.streaming_duration is None
        assert response.tokens_per_second is None

    def test_from_raw_response_with_streaming_metrics(self) -> None:
        """Test creating response from raw data with streaming metrics."""
        raw_data = {
            "response": "Test response",
            "conversation_id": "conv123",
            "input_tokens": 50,
            "output_tokens": 150,
            "time_to_first_token": 0.234,
            "streaming_duration": 3.456,
            "tokens_per_second": 46.5,
        }

        response = APIResponse.from_raw_response(raw_data)

        assert response.input_tokens == 50
        assert response.output_tokens == 150
        assert response.time_to_first_token == 0.234
        assert response.streaming_duration == 3.456
        assert response.tokens_per_second == 46.5

    def test_from_raw_response_without_streaming_metrics(self) -> None:
        """Test creating response from raw data without streaming metrics (query endpoint)."""
        raw_data = {
            "response": "Test response",
            "conversation_id": "conv123",
            "input_tokens": 50,
            "output_tokens": 150,
        }

        response = APIResponse.from_raw_response(raw_data)

        assert response.input_tokens == 50
        assert response.output_tokens == 150
        assert response.time_to_first_token is None
        assert response.streaming_duration is None
        assert response.tokens_per_second is None
