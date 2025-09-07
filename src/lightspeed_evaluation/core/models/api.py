"""API communication data models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .data import ToolCallData
from ..constants import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER


class AttachmentData(BaseModel):
    """Individual attachment structure for API."""

    attachment_type: str = Field(default="configuration", description="Attachment type")
    content: str = Field(..., description="Attachment content or reference")
    content_type: str = Field(default="text/plain", description="Content type")


class APIRequest(BaseModel):
    """API request model for dynamic data generation."""

    query: str = Field(..., min_length=1, description="User query")
    provider: str = Field(default=DEFAULT_LLM_PROVIDER, description="LLM provider")
    model: str = Field(default=DEFAULT_LLM_MODEL, description="LLM model")
    no_tools: bool = Field(default=False, description="Disable tool usage")
    conversation_id: Optional[str] = Field(
        default=None, description="Conversation ID for context"
    )
    system_prompt: Optional[str] = Field(
        default=None, description="System prompt override"
    )
    attachments: Optional[List[AttachmentData]] = Field(
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
        provider = kwargs.get("provider", DEFAULT_LLM_PROVIDER)
        model = kwargs.get("model", DEFAULT_LLM_MODEL)
        no_tools = kwargs.get("no_tools", False)
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


class APIResponse(BaseModel):
    """API response model."""

    response: str = Field(..., description="Response text from API")
    conversation_id: str = Field(
        ..., min_length=1, description="Conversation tracking ID"
    )
    tool_calls: List[List[ToolCallData]] = Field(
        default_factory=list, description="Tool call sequences"
    )
    contexts: List[str] = Field(
        default_factory=list, description="Context from API"
    )

    @classmethod
    def from_raw_response(cls, raw_data: Dict[str, Any]) -> "APIResponse":
        """Create APIResponse from raw API response data."""
        # Convert tool calls from List[List[Dict]] to List[List[ToolCallData]]
        tool_call_sequences = []
        raw_tool_calls = raw_data.get("tool_calls", [])

        for sequence in raw_tool_calls:
            tool_call_sequence = []
            for tool_call_dict in sequence:
                tool_call_sequence.append(
                    ToolCallData(
                        name=tool_call_dict.get("name", ""),
                        arguments=tool_call_dict.get("arguments", {}),
                    )
                )
            tool_call_sequences.append(tool_call_sequence)

        return cls(
            response=raw_data["response"].strip(),
            conversation_id=raw_data.get("conversation_id"),
            tool_calls=tool_call_sequences,
            contexts=raw_data.get("contexts", []),  # Contexts from API output
        )
