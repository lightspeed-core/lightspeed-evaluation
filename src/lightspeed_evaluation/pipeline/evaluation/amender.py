"""API Data Amendment module - handles API data enrichment."""

import logging
from typing import Any, Optional

from lightspeed_evaluation.core.api import APIClient
from lightspeed_evaluation.core.models import EvaluationData
from lightspeed_evaluation.core.system.exceptions import APIError

logger = logging.getLogger(__name__)


class APIDataAmender:
    """Handles API data amendment for evaluation data."""

    def __init__(self, api_client: Optional[APIClient]):
        """Initialize with API client."""
        self.api_client = api_client

    def amend_conversation_data(self, conv_data: EvaluationData) -> Optional[str]:
        """Amend conversation data with API responses.

        Returns:
            Optional[str]: Error message if any API error occurred, None if successful
        """
        if not self.api_client:
            return None

        # Track conversation_id across turns
        conversation_id: Optional[str] = None

        for turn_data in conv_data.turns:
            logger.debug("Amending turn %s with API data", turn_data.turn_id)

            try:
                api_response = self.api_client.query(
                    query=turn_data.query,
                    conversation_id=conversation_id,
                    attachments=turn_data.attachments,
                )
                conversation_id = api_response.conversation_id  # Track for next turns

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

                logger.debug("Data amended for turn %s", turn_data.turn_id)

            except APIError as e:
                error_msg = f"API Error for turn {turn_data.turn_id}: {e}"
                logger.error(error_msg)
                return error_msg
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_msg = f"Unexpected error for turn {turn_data.turn_id}: {e}"
                logger.error(error_msg)
                return error_msg

        return None  # No errors occurred

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
