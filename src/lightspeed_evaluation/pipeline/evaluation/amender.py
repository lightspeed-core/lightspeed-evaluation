"""API Data Amendment module - handles API data enrichment."""

import logging
from typing import Any, Dict, Optional

from ...core.api import APIClient
from ...core.models import EvaluationData
from ...core.system.exceptions import APIError

logger = logging.getLogger(__name__)


class APIDataAmender:
    """Handles API data amendment for evaluation data."""

    def __init__(self, api_client: Optional[APIClient]):
        """Initialize with API client."""
        self.api_client = api_client

    def amend_conversation_data(self, conv_data: EvaluationData) -> bool:
        """Amend conversation data with API responses.

        Returns:
            bool: True if any API error occurred, False if successful
        """
        if not self.api_client:
            return False

        # Track conversation_id across turns
        conversation_id: Optional[str] = None

        for turn_data in conv_data.turns:
            # Use existing conversation_id if available
            if turn_data.conversation_id:
                conversation_id = turn_data.conversation_id

            logger.debug("Amending turn %s with API data", turn_data.turn_id)

            try:
                api_response = self.api_client.query(
                    query=turn_data.query,
                    conversation_id=conversation_id,
                    attachments=turn_data.attachments,
                )
                conversation_id = api_response.conversation_id  # Track for next turns

                # UPDATE ORIGINAL YAML DATA: This modifies the loaded TurnData object in-place
                # Update response from API
                turn_data.response = api_response.response
                turn_data.conversation_id = api_response.conversation_id

                # Update contexts from API output
                if api_response.contexts:
                    turn_data.contexts = api_response.contexts

                # Update tool calls from API output (already converted to ToolCallData)
                if api_response.tool_calls:
                    logger.debug(
                        "Tool calls provided: %d sequences", len(api_response.tool_calls)
                    )
                    turn_data.tool_calls = api_response.tool_calls

                logger.debug("Data amended for turn %s", turn_data.turn_id)

            except APIError as e:
                logger.error("API Error for turn %s: %s", turn_data.turn_id, e)
                return True  # Indicate API error occurred
            except Exception as e:
                logger.error("Unexpected error for turn %s: %s", turn_data.turn_id, e)
                return True  # Indicate error occurred

        return False  # No errors occurred

    def get_amendment_summary(self, conv_data: EvaluationData) -> Dict[str, Any]:
        """Get summary of what would be amended for a conversation."""
        return {
            "conversation_id": conv_data.conversation_group_id,
            "total_turns": len(conv_data.turns),
            "api_enabled": self.api_client is not None,
            "turns_with_existing_data": len(
                [turn for turn in conv_data.turns if turn.response or turn.tool_calls]
            ),
        }
