"""HTTP client for agent API communication."""

import json
import logging
import os
from typing import Any, Optional

import httpx

from .exceptions import AgentAPIError
from .streaming_parser import parse_streaming_response

logger = logging.getLogger(__name__)


class AgentHttpClient:
    """HTTP client for agent API communication."""

    def __init__(  # pylint: disable=R0913,R0917
        self,
        endpoint: str,
        version: str = "v1",
        token_file: Optional[str] = None,
        verify_ssl: bool = True,
        timeout: float = 300.0,
    ):
        """Initialize HTTP client.

        Args:
            endpoint: Base API URL.
            version: API version (e.g., v1, v2). Defaults to "v1".
            token_file: Optional path to token file for authentication.
            verify_ssl: Whether to verify SSL certificates. Defaults to True.
            timeout: Default timeout in seconds for requests. Defaults to 300.0.
        """
        self.endpoint = endpoint
        self.version = version
        self.client: Optional[httpx.Client] = None
        self._setup_client(token_file, verify_ssl, timeout)

    def _setup_client(
        self, token_file: Optional[str], verify_ssl: bool, timeout: float
    ) -> None:
        """Initialize HTTP client with authentication."""
        try:
            self.client = httpx.Client(
                base_url=self.endpoint, verify=verify_ssl, timeout=timeout
            )

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
    ) -> dict[str, Any]:
        """Query the agent using non-streaming endpoint."""
        if not self.client:
            raise AgentAPIError("HTTP client not initialized")

        try:
            response = self.client.post(
                f"/{self.version}/query",
                json=api_input,
                timeout=timeout,
            )
            response.raise_for_status()

            response_data = response.json()
            if "response" not in response_data:
                raise AgentAPIError("Agent response missing 'response' field")

            agent_response = response_data["response"].strip()
            conversation_id = response_data.get("conversation_id", "").strip()
            tool_calls = response_data.get("tool_calls", [])

            # Format tool calls to match expected structure (list of sequences)
            formatted_tool_calls = self._format_query_endpoint_tool_calls(tool_calls)

            return {
                "response": agent_response,
                "conversation_id": conversation_id,
                "tool_calls": formatted_tool_calls,
            }

        except httpx.TimeoutException as e:
            raise AgentAPIError(f"Agent query timeout after {timeout} seconds") from e
        except httpx.HTTPStatusError as e:
            raise AgentAPIError(
                f"Agent API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except Exception as e:
            raise AgentAPIError(f"Unexpected error querying agent: {e}") from e

    def _format_query_endpoint_tool_calls(
        self, tool_calls: list
    ) -> list[list[dict[str, Any]]]:
        """Format tool calls from query endpoint to match expected structure."""
        if not tool_calls:
            return []

        formatted_sequences = []
        for tool_call in tool_calls:
            # OLS dependency
            formatted_tool = {
                "tool_name": tool_call.get("name", ""),
                "arguments": tool_call.get("args", {}),
            }
            formatted_sequences.append([formatted_tool])

        return formatted_sequences

    def streaming_query_agent(
        self, api_input: dict[str, str], timeout: int = 300
    ) -> dict[str, Any]:
        """Query the agent using streaming endpoint."""
        if not self.client:
            raise AgentAPIError("HTTP client not initialized")

        try:
            with self.client.stream(
                "POST",
                f"/{self.version}/streaming_query",
                json=api_input,
                timeout=timeout,
            ) as response:
                # Potential change lsc-stack to provide SSE error message
                if response.status_code != 200:
                    error_content = response.read().decode("utf-8")
                    try:
                        error_data = json.loads(error_content)
                        if isinstance(error_data, dict) and "detail" in error_data:
                            detail = error_data["detail"]
                            if isinstance(detail, dict):
                                response_msg = detail.get("response", "")
                                cause_msg = detail.get("cause", "")
                                error_msg = (
                                    f"{response_msg} - {cause_msg}"
                                    if cause_msg
                                    else response_msg
                                )
                            else:
                                error_msg = str(detail)
                        else:
                            error_msg = error_content
                    except (json.JSONDecodeError, KeyError, TypeError):
                        error_msg = error_content

                    raise httpx.HTTPStatusError(
                        message=f"Agent API error: {response.status_code} - {error_msg}",
                        request=response.request,
                        response=response,
                    )

                return parse_streaming_response(response)

        except httpx.TimeoutException as e:
            raise AgentAPIError(
                f"Agent streaming query timeout after {timeout} seconds"
            ) from e
        except httpx.HTTPStatusError as e:
            raise AgentAPIError(str(e)) from e
        except ValueError as e:
            raise AgentAPIError(f"Streaming response validation error: {e}") from e
        except Exception as e:
            raise AgentAPIError(f"Unexpected error in streaming query: {e}") from e

    def close(self) -> None:
        """Close HTTP client."""
        if self.client:
            self.client.close()
