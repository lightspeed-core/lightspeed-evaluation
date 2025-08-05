"""HTTP client for agent API communication."""

import json
import logging
import os
from typing import Optional

import httpx

from .exceptions import AgentAPIError

logger = logging.getLogger(__name__)


class AgentHttpClient:
    """HTTP client for agent API communication."""

    def __init__(self, endpoint: str, token_file: Optional[str] = None):
        """Initialize HTTP client."""
        self.endpoint = endpoint
        self.client: Optional[httpx.Client] = None
        self._setup_client(token_file)

    def _setup_client(self, token_file: Optional[str]) -> None:
        """Initialize HTTP client with authentication."""
        try:
            # enable verify, currently for eval it is set to False
            self.client = httpx.Client(base_url=self.endpoint, verify=False)

            token = None
            if token_file:
                token = self._read_token_file(token_file)
            token = token or os.getenv("AGENT_API_TOKEN")
            if token and self.client:
                self.client.headers.update({"Authorization": f"Bearer {token}"})

        except Exception as e:
            raise AgentAPIError(f"Failed to setup HTTP client: {e}") from e

    def _read_token_file(self, token_file: str) -> str:
        """Read authentication token from file."""
        try:
            with open(token_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError as e:
            raise AgentAPIError(f"Token file not found: {token_file}") from e
        except Exception as e:
            raise AgentAPIError(f"Error reading token file: {e}") from e

    def query_agent(
        self, api_input: dict[str, str], timeout: int = 300
    ) -> tuple[str, str]:
        """Query the agent and return response."""
        if not self.client:
            raise AgentAPIError("HTTP client not initialized")

        try:
            response = self.client.post(
                "/v1/query",
                json=api_input,
                timeout=timeout,
            )
            response.raise_for_status()

            response_data = response.json()
            if "response" not in response_data:
                raise AgentAPIError("Agent response missing 'response' field")
            agent_response = response_data["response"].strip()

            conversation_id = response_data.get("conversation_id", "").strip()

            return agent_response, conversation_id

        except httpx.TimeoutException as e:
            raise AgentAPIError(f"Agent query timeout after {timeout} seconds") from e
        except httpx.HTTPStatusError as e:
            raise AgentAPIError(
                f"Agent API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise AgentAPIError(f"Unexpected error querying agent: {e}") from e

    def streaming_query_agent(
        self, api_input: dict[str, str], timeout: int = 300
    ) -> tuple[str, str]:
        """Query the agent using streaming endpoint and return response."""
        if not self.client:
            raise AgentAPIError("HTTP client not initialized")

        try:
            # Response format is as per the lightspeed-stack
            with self.client.stream(
                "POST",
                "/v1/streaming_query",
                json=api_input,
                timeout=timeout,
            ) as response:
                response.raise_for_status()

                conversation_id = ""
                final_response = ""

                for line in response.iter_lines():
                    line = line.strip()

                    # Skip empty lines or non-data lines
                    if not line or not line.startswith("data: "):
                        continue

                    # Remove "data: " prefix to get JSON content
                    json_data = line[6:]

                    try:
                        data = json.loads(json_data)
                        event = data.get("event", "")
                        event_data = data.get("data", {})

                        # Extract conversation_id from start event
                        if event == "start" and "conversation_id" in event_data:
                            conversation_id = event_data["conversation_id"].strip()
                            logger.debug("Found conversation_id: %s", conversation_id)

                        # Extract final response from turn_complete event
                        elif event == "turn_complete" and "token" in event_data:
                            final_response = event_data["token"].strip()
                            logger.debug(
                                "Found final response (%d characters)",
                                len(final_response),
                            )

                    except json.JSONDecodeError:
                        logger.debug(
                            "Failed to parse JSON from streaming response: %s",
                            json_data,
                        )
                        continue

                if not final_response:
                    raise AgentAPIError("No final response found in streaming output")
                if not conversation_id:
                    raise AgentAPIError("No Conversation ID found")

                return final_response, conversation_id

        except httpx.TimeoutException as e:
            raise AgentAPIError(
                f"Agent streaming query timeout after {timeout} seconds"
            ) from e
        except httpx.HTTPStatusError as e:
            raise AgentAPIError(
                f"Agent API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise AgentAPIError(f"Unexpected error in streaming query: {e}") from e

    def close(self) -> None:
        """Close HTTP client."""
        if self.client:
            self.client.close()
