"""API Data Amendment module - handles API data enrichment."""

import logging
import time
from typing import Any, Optional

from lightspeed_evaluation.core.api import APIClient
from lightspeed_evaluation.core.models import EvaluationData, TurnData
from lightspeed_evaluation.core.system.exceptions import APIError

logger = logging.getLogger(__name__)


class APIDataAmender:
    """Handles API data amendment for evaluation data."""

    def __init__(self, api_client: Optional[APIClient]):
        """Initialize with API client."""
        self.api_client = api_client

    def amend_single_turn(
        self, turn_data: TurnData, conversation_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Amend single turn data with API response.

        Args:
            turn_data: The turn data to amend
            conversation_id: Optional conversation ID from previous turns

        Returns:
            tuple: (error_message, updated_conversation_id)
                - error_message: None if successful, error string if failed
                - updated_conversation_id: The conversation ID for next turns
        """
        if not self.api_client:
            return None, conversation_id

        logger.debug("Amending turn %s with API data", turn_data.turn_id)

        # Track API call execution time
        api_start_time = time.perf_counter()
        try:
            api_response = self.api_client.query(
                query=turn_data.query or "",
                conversation_id=conversation_id,
                attachments=turn_data.attachments,
                extra_request_params=turn_data.extra_request_params,
            )
            api_latency = time.perf_counter() - api_start_time

            # AMEND EVALUATION DATA: This modifies the loaded TurnData object in-place
            # Update response from API
            turn_data.response = api_response.response
            turn_data.conversation_id = api_response.conversation_id

            # Update contexts from API output
            if api_response.contexts:
                turn_data.contexts = api_response.contexts

            # Update tool calls from API output
            if api_response.tool_calls:
                logger.debug(
                    "Tool calls provided: %d sequences",
                    len(api_response.tool_calls),
                )
                turn_data.tool_calls = api_response.tool_calls
            # Update token usage from API output (with fallback to 0 if not present)
            turn_data.api_input_tokens = getattr(api_response, "input_tokens", 0)
            turn_data.api_output_tokens = getattr(api_response, "output_tokens", 0)

            # Update API latency only for actual API calls (cached responses have 0 tokens)
            turn_data.agent_latency = (
                api_latency
                if (turn_data.api_input_tokens > 0 or turn_data.api_output_tokens > 0)
                else 0.0
            )
            logger.debug(
                "Token usage for turn %s: input=%d, output=%d, API latency=%.3fs",
                turn_data.turn_id,
                turn_data.api_input_tokens,
                turn_data.api_output_tokens,
                turn_data.agent_latency,
            )

            # Update streaming performance metrics (only available for streaming endpoint)
            turn_data.time_to_first_token = getattr(
                api_response, "time_to_first_token", None
            )
            turn_data.streaming_duration = getattr(
                api_response, "streaming_duration", None
            )
            turn_data.tokens_per_second = getattr(
                api_response, "tokens_per_second", None
            )
            if turn_data.time_to_first_token is not None:
                # Format metrics individually to preserve None values for edge cases
                duration_str = (
                    f"{turn_data.streaming_duration:.3f}s"
                    if turn_data.streaming_duration is not None
                    else "N/A"
                )
                throughput_str = (
                    f"{turn_data.tokens_per_second:.2f} tokens/sec"
                    if turn_data.tokens_per_second is not None
                    else "N/A"
                )
                logger.debug(
                    "Streaming metrics for turn %s: TTFT=%.3fs, duration=%s, "
                    "throughput=%s",
                    turn_data.turn_id,
                    turn_data.time_to_first_token,
                    duration_str,
                    throughput_str,
                )

            logger.debug("Data amended for turn %s", turn_data.turn_id)
            return None, api_response.conversation_id

        except APIError as e:
            # Record elapsed time even on error
            api_latency = time.perf_counter() - api_start_time
            turn_data.agent_latency = api_latency
            error_msg = f"API Error for turn {turn_data.turn_id}: {e}"
            logger.error(error_msg)
            return error_msg, conversation_id

    def get_amendment_summary(self, conv_data: EvaluationData) -> dict[str, Any]:
        """Get summary of what would be amended for a conversation."""
        return {
            "conversation_group_id": conv_data.conversation_group_id,
            "total_turns": len(conv_data.turns),
            "api_enabled": self.api_client is not None,
            "turns_with_existing_data": len(
                [turn for turn in conv_data.turns if turn.response or turn.tool_calls]
            ),
        }
