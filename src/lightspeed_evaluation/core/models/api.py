"""API communication data models."""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from lightspeed_evaluation.core.models.mixins import StreamingMetricsMixin


class RAGChunk(BaseModel):
    """RAG chunk data from lightspeed-stack API."""

    model_config = ConfigDict(extra="forbid")

    content: str = Field(..., description="RAG chunk content")
    source: str = Field(..., description="Source of the RAG chunk")
    score: Optional[float] = Field(
        default=None, description="Relevance score of the chunk"
    )


class AttachmentData(BaseModel):
    """Individual attachment structure for API."""

    model_config = ConfigDict(extra="forbid")

    attachment_type: str = Field(default="configuration", description="Attachment type")
    content: str = Field(..., description="Attachment content or reference")
    content_type: str = Field(default="text/plain", description="Content type")


class APIRequest(BaseModel):
    """API request model for dynamic data generation."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="User query")
    provider: Optional[str] = Field(default=None, description="LLM provider")
    model: Optional[str] = Field(default=None, description="LLM model")
    no_tools: Optional[bool] = Field(default=None, description="Disable tool usage")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for context"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt override"
    )
    attachments: Optional[list[AttachmentData]] = Field(
        default=None, description="File attachments"
    )

    @classmethod
    def create(
        cls,
        query: str,
        **kwargs: Any,
    ) -> "APIRequest":
        """Create API request with optional attachments."""
        # Extract parameters with defaults
        provider = kwargs.get("provider")
        model = kwargs.get("model")
        no_tools = kwargs.get("no_tools")
        conversation_id = kwargs.get("conversation_id")
        system_prompt = kwargs.get("system_prompt")
        attachments = kwargs.get("attachments")
        attachment_data = None
        if attachments:
            attachment_data = [
                AttachmentData(content=attachment) for attachment in attachments
            ]

        return cls(
            query=query,
            provider=provider,
            model=model,
            no_tools=no_tools,
            conversation_id=conversation_id,
            system_prompt=system_prompt,
            attachments=attachment_data,
        )


class APIResponse(StreamingMetricsMixin):
    """API response model."""

    model_config = ConfigDict(extra="forbid")

    response: str = Field(..., description="Response text from API")
    conversation_id: str = Field(
        ..., min_length=1, description="Conversation tracking ID"
    )
    tool_calls: list[list[dict[str, Any]]] = Field(
        default_factory=list, description="Tool call sequences"
    )
    contexts: list[str] = Field(default_factory=list, description="Context from API")
    input_tokens: int = Field(default=0, ge=0, description="Input tokens used")
    output_tokens: int = Field(default=0, ge=0, description="Output tokens used")

    @classmethod
    def from_raw_response(cls, raw_data: dict[str, Any]) -> "APIResponse":
        """Create APIResponse from raw API response data."""
        tool_call_sequences = raw_data.get("tool_calls", [])

        conversation_id = raw_data.get("conversation_id")
        if not conversation_id:
            raise ValueError("conversation_id is required in API response")

        # Extract contexts from RAG chunks
        raw_rag_chunks = raw_data.get("rag_chunks", [])
        contexts = []
        if raw_rag_chunks:
            for chunk_data in raw_rag_chunks:
                if isinstance(chunk_data, dict) and "content" in chunk_data:
                    contexts.append(chunk_data["content"])

        # Extract token counts
        input_tokens = raw_data.get("input_tokens", 0)
        output_tokens = raw_data.get("output_tokens", 0)

        # Extract streaming performance metrics (only available for streaming endpoint)
        time_to_first_token = raw_data.get("time_to_first_token")
        streaming_duration = raw_data.get("streaming_duration")
        tokens_per_second = raw_data.get("tokens_per_second")

        return cls(
            response=raw_data["response"].strip(),
            conversation_id=conversation_id,
            tool_calls=tool_call_sequences,
            contexts=contexts,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            time_to_first_token=time_to_first_token,
            streaming_duration=streaming_duration,
            tokens_per_second=tokens_per_second,
        )
