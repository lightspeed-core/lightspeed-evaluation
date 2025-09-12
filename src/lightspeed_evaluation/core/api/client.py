"""API client for actual data generation."""

import json
import logging
import os
from typing import Any, Optional

import httpx

from ..constants import (
    DEFAULT_API_TIMEOUT,
    DEFAULT_ENDPOINT_TYPE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    SUPPORTED_ENDPOINT_TYPES,
)
from ..models import APIRequest, APIResponse
from ..system.exceptions import APIError
from .streaming_parser import parse_streaming_response

logger = logging.getLogger(__name__)


class APIClient:
    """API client for actual data generation."""

    def __init__(
        self,
        api_base: str,
        *,
        config: Optional[dict[str, Any]] = None,
        endpoint_type: str = DEFAULT_ENDPOINT_TYPE,
        timeout: int = DEFAULT_API_TIMEOUT,
    ):
        """Initialize the client with configuration."""
        self.api_base = api_base
        self.endpoint_type = endpoint_type
        self.timeout = timeout
        # LLM configuration with defaults
        default_config = {
            "provider": DEFAULT_LLM_PROVIDER,
            "model": DEFAULT_LLM_MODEL,
            "no_tools": None,
            "system_prompt": None,
        }
        self.llm_config = {**default_config, **(config or {})}

        self.client: Optional[httpx.Client] = None

        self._validate_endpoint_type()
        self._setup_client()

    def _validate_endpoint_type(self) -> None:
        """Validate endpoint type is supported."""
        if self.endpoint_type not in SUPPORTED_ENDPOINT_TYPES:
            raise APIError(
                f"Unsupported endpoint type: {self.endpoint_type}. "
                f"Must be one of {SUPPORTED_ENDPOINT_TYPES}"
            )

    def _setup_client(self) -> None:
        """Initialize API client with authentication."""
        try:
            # Enable verify, currently for eval it is set to False
            verify = False
            self.client = httpx.Client(
                base_url=self.api_base, verify=verify, timeout=self.timeout
            )
            self.client.headers.update({"Content-Type": "application/json"})

            # Use API_KEY environment variable for authentication
            api_key = os.getenv("API_KEY")
            if api_key and self.client:
                self.client.headers.update({"Authorization": f"Bearer {api_key}"})

        except Exception as e:
            raise APIError(f"Failed to setup API client: {e}") from e

    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,
    ) -> APIResponse:
        """Query the API using the configured endpoint type.

        Args:
            query: The question/query to ask
            conversation_id: Optional conversation ID for context
            attachments: Optional list of attachments

        Returns:
            APIResponse with Response, Tool calls, Conversation ID
        """
        if not self.client:
            raise APIError("API client not initialized")

        api_request = self._prepare_request(query, conversation_id, attachments)

        if self.endpoint_type == "streaming":
            return self._streaming_query(api_request)
        return self._standard_query(api_request)

    def _prepare_request(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,
    ) -> APIRequest:
        """Prepare API request with common parameters."""
        return APIRequest.create(
            query=query,
            provider=self.llm_config["provider"],
            model=self.llm_config["model"],
            no_tools=self.llm_config["no_tools"],
            conversation_id=conversation_id,
            system_prompt=self.llm_config["system_prompt"],
            attachments=attachments,
        )

    def _standard_query(self, api_request: APIRequest) -> APIResponse:
        """Query the API using non-streaming endpoint."""
        if not self.client:
            raise APIError("HTTP client not initialized")
        try:
            response = self.client.post(
                "/v1/query",
                json=api_request.model_dump(exclude_none=True),
            )
            response.raise_for_status()

            response_data = response.json()
            if "response" not in response_data:
                raise APIError("API response missing 'response' field")

            # Format tool calls to match streaming endpoint format
            # Currently only compatible with OLS
            if "tool_calls" in response_data and response_data["tool_calls"]:
                raw_tool_calls = response_data["tool_calls"]
                formatted_tool_calls = []

                # Convert list[dict] to list[list[dict]] format
                for tool_call in raw_tool_calls:
                    if isinstance(tool_call, dict):
                        formatted_tool = {
                            "tool_name": tool_call.get("name", ""),
                            "arguments": tool_call.get("args", {}),
                        }
                        formatted_tool_calls.append([formatted_tool])

                response_data["tool_calls"] = formatted_tool_calls

            return APIResponse.from_raw_response(response_data)

        except httpx.TimeoutException as e:
            raise self._handle_timeout_error("standard", self.timeout) from e
        except httpx.HTTPStatusError as e:
            raise self._handle_http_error(e) from e
        except ValueError as e:
            raise self._handle_validation_error(e) from e
        except APIError:
            raise
        except Exception as e:
            raise self._handle_unexpected_error(e, "standard query") from e

    def _streaming_query(self, api_request: APIRequest) -> APIResponse:
        """Query the API using streaming endpoint."""
        if not self.client:
            raise APIError("HTTP client not initialized")
        try:
            with self.client.stream(
                "POST",
                "/v1/streaming_query",
                json=api_request.model_dump(exclude_none=True),
            ) as response:
                self._handle_response_errors(response)
                raw_data = parse_streaming_response(response)
                return APIResponse.from_raw_response(raw_data)

        except httpx.TimeoutException as e:
            raise self._handle_timeout_error("streaming", self.timeout) from e
        except httpx.HTTPStatusError as e:
            raise APIError(str(e)) from e
        except ValueError as e:
            raise self._handle_validation_error(e) from e
        except APIError:
            raise
        except Exception as e:
            raise self._handle_unexpected_error(e, "streaming query") from e

    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Handle HTTP response errors for streaming endpoint."""
        if response.status_code != 200:
            error_msg = self._extract_error_message(response)
            raise httpx.HTTPStatusError(
                message=f"Agent API error: {response.status_code} - {error_msg}",
                request=response.request,
                response=response,
            )

    def _extract_error_message(self, response: httpx.Response) -> str:
        """Extract error message from response."""
        try:
            error_content = response.read().decode("utf-8")
            error_data = json.loads(error_content)

            if isinstance(error_data, dict) and "detail" in error_data:
                detail = error_data["detail"]
                if isinstance(detail, dict):
                    response_msg = detail.get("response", "")
                    cause_msg = detail.get("cause", "")
                    return (
                        f"{response_msg} - {cause_msg}" if cause_msg else response_msg
                    )
                return str(detail)
            return error_content
        except (json.JSONDecodeError, KeyError, TypeError):
            return (
                response.read().decode("utf-8")
                if hasattr(response, "read")
                else "Unknown error"
            )

    def _handle_timeout_error(self, endpoint_type: str, timeout: int) -> APIError:
        """Create appropriate timeout error message."""
        return APIError(f"API {endpoint_type} query timeout after {timeout} seconds")

    def _handle_http_error(self, e: httpx.HTTPStatusError) -> APIError:
        """Handle HTTP status errors."""
        return APIError(f"API error: {e.response.status_code} - {e.response.text}")

    def _handle_validation_error(self, e: ValueError) -> APIError:
        """Handle validation errors."""
        return APIError(f"Response validation error: {e}")

    def _handle_unexpected_error(self, e: Exception, operation: str) -> APIError:
        """Handle unexpected errors."""
        return APIError(f"Unexpected error in {operation}: {e}")

    def close(self) -> None:
        """Close API client."""
        if self.client:
            self.client.close()
