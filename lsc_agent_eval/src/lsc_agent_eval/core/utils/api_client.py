"""HTTP client for agent API communication."""

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
        self, query: str, provider: str, model: str, timeout: int = 300
    ) -> str:
        """Query the agent and return response."""
        if not self.client:
            raise AgentAPIError("HTTP client not initialized")

        try:
            api_input = {
                "query": query,
                "provider": provider,
                "model": model,
            }
            response = self.client.post(
                "/v1/query",
                json=api_input,
                timeout=timeout,
            )
            response.raise_for_status()

            response_data = response.json()
            if "response" not in response_data:
                raise AgentAPIError("Agent response missing 'response' field")

            return response_data["response"].strip()

        except httpx.TimeoutException as e:
            raise AgentAPIError(f"Agent query timeout after {timeout} seconds") from e
        except httpx.HTTPStatusError as e:
            raise AgentAPIError(
                f"Agent API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise AgentAPIError(f"Unexpected error querying agent: {e}") from e

    def close(self) -> None:
        """Close HTTP client."""
        if self.client:
            self.client.close()
