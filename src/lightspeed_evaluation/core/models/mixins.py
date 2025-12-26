"""Shared model mixins for the evaluation framework."""

from typing import Optional

from pydantic import BaseModel, Field


class StreamingMetricsMixin(BaseModel):
    """Mixin providing streaming performance metric fields.

    These fields capture performance metrics when using the streaming query endpoint:
    - time_to_first_token: Time from request start to first content token
    - streaming_duration: Total time to receive all tokens
    - tokens_per_second: Output throughput calculation

    These fields are None when using the non-streaming query endpoint.
    """

    time_to_first_token: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time to first token in seconds (streaming only)",
    )
    streaming_duration: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total streaming duration in seconds (streaming only)",
    )
    tokens_per_second: Optional[float] = Field(
        default=None,
        ge=0,
        description="Output tokens per second throughput (streaming only)",
    )
