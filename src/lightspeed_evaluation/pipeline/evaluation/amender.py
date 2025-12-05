"""API Data Amendment module - handles API data enrichment."""

import logging
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

        try:
            api_response = self.api_client.query(
                query=turn_data.query,
                conversation_id=conversation_id,
                attachments=turn_data.attachments,
            )

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
            logger.debug(
                "Token usage for turn %s: input=%d, output=%d",
                turn_data.turn_id,
                turn_data.api_input_tokens,
                turn_data.api_output_tokens,
            )

            logger.debug("Data amended for turn %s", turn_data.turn_id)
            return None, api_response.conversation_id

        except APIError as e:
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
