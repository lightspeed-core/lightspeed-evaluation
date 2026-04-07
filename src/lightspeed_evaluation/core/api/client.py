"""API client for actual data generation."""

import hashlib
import json
import logging
import os
from typing import Any, Optional, cast

import httpx
from diskcache import Cache
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    RetryError,
)

from lightspeed_evaluation.core.api.streaming_parser import parse_streaming_response
from lightspeed_evaluation.core.constants import (
    SUPPORTED_ENDPOINT_TYPES,
)
from lightspeed_evaluation.core.models import APIConfig, APIRequest, APIResponse
from lightspeed_evaluation.core.system.exceptions import APIError

logger = logging.getLogger(__name__)


def _is_too_many_requests_error(exception: BaseException) -> bool:
    """Check if exception is a 429 error."""
    return (
        isinstance(exception, httpx.HTTPStatusError)
        and exception.response.status_code == 429
    )


class APIClient:
    """API client for actual data generation."""

    def __init__(
        self,
        config: APIConfig,
    ):
        """Initialize the client with configuration."""
        self.config = config

        self.client: Optional[httpx.Client] = None

        cache = None
        if config.cache_enabled:
            cache = Cache(config.cache_dir)
        self.cache = cache

        self._validate_endpoint_type()
        self._setup_client()

        # Wrap methods with retry decorator for handling 429 Too Many Requests errors
        retry_decorator = self._create_retry_decorator()
        self._standard_query_with_retry = retry_decorator(self._standard_query)
        self._streaming_query_with_retry = retry_decorator(self._streaming_query)

    def _create_retry_decorator(self) -> Any:
        return retry(
            retry=retry_if_exception(_is_too_many_requests_error),
            stop=stop_after_attempt(
                self.config.num_retries + 1
            ),  # +1 to account for the initial attempt
            wait=wait_exponential(multiplier=1, min=4, max=60),  # multiplier * 2^x
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=False,  # If all retry attempts are exhausted, RetryError is raised
        )

    def _validate_endpoint_type(self) -> None:
        """Validate endpoint type is supported."""
        if self.config.endpoint_type not in SUPPORTED_ENDPOINT_TYPES:
            raise APIError(
                f"Unsupported endpoint type: {self.config.endpoint_type}. "
                f"Must be one of {SUPPORTED_ENDPOINT_TYPES}"
            )

    def _setup_client(self) -> None:
        """Initialize API client with authentication."""
        try:
            # Enable verify, currently for eval it is set to False
            verify = False
            self.client = httpx.Client(
                base_url=self.config.api_base,
                verify=verify,
                timeout=self.config.timeout,
            )
            self.client.headers.update({"Content-Type": "application/json"})

            # Set up MCP headers based on configuration
            mcp_headers_dict = self._build_mcp_headers()
            if mcp_headers_dict:
                self.client.headers.update(
                    {"MCP-HEADERS": json.dumps(mcp_headers_dict)}
                )
            else:
                # Use API_KEY environment variable for authentication (backward compatibility)
                api_key = os.getenv("API_KEY")
                if api_key and self.client:
                    self.client.headers.update({"Authorization": f"Bearer {api_key}"})

        except Exception as e:
            raise APIError(f"Failed to setup API client: {e}") from e

    def _build_mcp_headers(self) -> dict[str, dict[str, str]]:
        """Build MCP headers based on configuration.

        Returns:
            Dictionary of MCP server headers, or empty dict if none configured.
        """
        # Use new MCP headers configuration if available and enabled
        if (
            self.config.mcp_headers
            and self.config.mcp_headers.enabled
            and self.config.mcp_headers.servers
        ):

            mcp_headers = {}
            for server_name, server_config in self.config.mcp_headers.servers.items():
                # Get token from environment variable
                token = os.getenv(server_config.env_var)
                if not token:
                    logger.warning(
                        "Environment variable '%s' not found "
                        "for MCP server '%s'. Skipping authentication.",
                        server_config.env_var,
                        server_name,
                    )
                    continue

                # Set up bearer auth headers
                header_name = server_config.header_name or "Authorization"
                header_value = f"Bearer {token}"

                mcp_headers[server_name] = {header_name: header_value}

            if not mcp_headers:
                # Fallback to API_KEY if MCP headers are enabled but no credentials found
                api_key = os.getenv("API_KEY")
                if not api_key:
                    raise APIError(
                        "MCP headers are enabled, but no valid server credentials were resolved "
                        "and no API_KEY environment variable is set. Check mcp_headers.servers "
                        "and required environment variables, or set API_KEY as fallback."
                    )
                # Return empty dict to signal fallback to API_KEY authentication
                return {}
            return mcp_headers

        # No MCP headers configured
        return {}

    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,
        extra_request_params: Optional[dict[str, Any]] = None,
    ) -> APIResponse:
        """Query the API using the configured endpoint type.

        Args:
            query: The question/query to ask
            conversation_id: Optional conversation ID for context
            attachments: Optional list of attachments
            extra_request_params: Optional per-turn extra params (overrides system defaults)

        Returns:
            APIResponse with Response, Tool calls, Conversation ID
        """
        if not self.client:
            raise APIError("API client not initialized")

        try:
            api_request = self._prepare_request(
                query, conversation_id, attachments, extra_request_params
            )
            if self.config.cache_enabled:
                cached_response = self._get_cached_response(api_request)
                if cached_response is not None:
                    logger.debug("Returning cached response for query: '%s'", query)
                    return cached_response

            if self.config.endpoint_type == "streaming":
                response = self._streaming_query_with_retry(api_request)
            else:
                response = self._standard_query_with_retry(api_request)

            if self.config.cache_enabled:
                self._add_response_to_cache(api_request, response)

            return response
        except RetryError as e:
            raise APIError(
                f"Maximum retry attempts ({self.config.num_retries}) reached "
                "due to persistent rate limiting (HTTP 429)."
            ) from e

    def _prepare_request(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,
        extra_request_params: Optional[dict[str, Any]] = None,
    ) -> APIRequest:
        """Prepare API request with common parameters."""
        # Merge extra params: system defaults, then per-turn overrides
        resolved_extra = {**(self.config.extra_request_params or {})}
        if extra_request_params:
            resolved_extra.update(extra_request_params)
        return APIRequest.create(
            query=query,
            provider=self.config.provider,
            model=self.config.model,
            no_tools=self.config.no_tools,
            conversation_id=conversation_id,
            system_prompt=self.config.system_prompt,
            attachments=attachments,
            extra_request_params=resolved_extra or None,
        )

    @staticmethod
    def _serialize_request(api_request: APIRequest) -> dict[str, Any]:
        """Serialize API request, flattening extra_request_params into the payload."""
        payload = api_request.model_dump(exclude_none=True)
        extra = payload.pop("extra_request_params", None)
        if extra:
            reserved = set(APIRequest.model_fields)
            for key, value in extra.items():
                if key not in reserved and key not in payload:
                    payload[key] = value
        return payload

    def _standard_query(self, api_request: APIRequest) -> APIResponse:
        """Query the API using non-streaming endpoint with retry on 429."""
        if not self.client:
            raise APIError("HTTP client not initialized")
        try:
            response = self.client.post(
                f"/{self.config.version}/query",
                json=self._serialize_request(api_request),
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
                        formatted_tool: dict[str, object] = {
                            "tool_name": tool_call.get("tool_name")
                            or tool_call.get("name")  # Current OLS
                            or "",
                            "arguments": tool_call.get("arguments")
                            or tool_call.get("args")  # Current OLS
                            or {},
                        }
                        # Capture tool result if present (optional field)
                        result = tool_call.get("result")
                        if result is not None:
                            formatted_tool["result"] = result
                        formatted_tool_calls.append([formatted_tool])

                response_data["tool_calls"] = formatted_tool_calls

            return APIResponse.from_raw_response(response_data)

        except httpx.TimeoutException as e:
            raise self._handle_timeout_error("standard", self.config.timeout) from e
        except httpx.HTTPStatusError as e:
            # Re-raise 429 errors without conversion to allow retry decorator to handle them
            if e.response.status_code == 429:
                raise
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
                f"/{self.config.version}/streaming_query",
                json=self._serialize_request(api_request),
            ) as response:
                self._handle_response_errors(response)
                raw_data = parse_streaming_response(response)
                return APIResponse.from_raw_response(raw_data)

        except httpx.TimeoutException as e:
            raise self._handle_timeout_error("streaming", self.config.timeout) from e
        except httpx.HTTPStatusError as e:
            # Re-raise 429 errors without conversion to allow retry decorator to handle them
            if e.response.status_code == 429:
                raise
            raise self._handle_http_error(e) from e
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

    def _get_cache_key(self, request: APIRequest) -> str:
        """Get cache key for the query."""
        # Note, python hash is initialized randomly so can't be used here
        request_dict = request.model_dump()
        keys_to_hash = [
            "query",
            "provider",
            "model",
            "no_tools",
            "system_prompt",
            "attachments",
            "extra_request_params",
        ]
        str_request = ",".join([str(request_dict[k]) for k in keys_to_hash])

        return hashlib.sha256(str_request.encode()).hexdigest()

    def _add_response_to_cache(
        self, request: APIRequest, response: APIResponse
    ) -> None:
        """Add answer to disk cache."""
        if self.cache is None:
            raise RuntimeError("cache is None, but used")
        key = self._get_cache_key(request)
        self.cache[key] = response

    def _get_cached_response(self, request: APIRequest) -> APIResponse | None:
        """Get answer from the disk cache."""
        if self.cache is None:
            raise RuntimeError("cache is None, but used")
        key = self._get_cache_key(request)
        cached_response = cast(APIResponse | None, self.cache.get(key))

        # Zero out token counts for cached responses since no API call was made
        if cached_response is not None:
            cached_response.input_tokens = 0
            cached_response.output_tokens = 0

        return cached_response

    def close(self) -> None:
        """Close API client."""
        if self.client:
            self.client.close()
