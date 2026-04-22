"""Pytest configuration and fixtures for api tests."""

from typing import Any

import pytest

from pytest_mock import MockerFixture
from lightspeed_evaluation.core.models import APIConfig


@pytest.fixture
def basic_api_config_query_endpoint() -> APIConfig:
    """Create test API config for query endpoint."""
    return APIConfig(
        enabled=True,
        api_base="http://localhost:8080",
        version="v1",
        endpoint_type="query",
        timeout=30,
        cache_enabled=False,
    )


@pytest.fixture
def basic_api_config_streaming_endpoint() -> APIConfig:
    """Create basic API configuration for streaming endpoint."""
    return APIConfig(
        enabled=True,
        api_base="http://localhost:8080",
        endpoint_type="streaming",
        timeout=30,
        provider="openai",
        model="gpt-4",
        cache_enabled=False,
    )


@pytest.fixture
def basic_api_config_infer_endpoint() -> APIConfig:
    """Create test API config for infer endpoint."""
    return APIConfig(
        enabled=True,
        api_base="http://localhost:8080",
        version="v1",
        endpoint_type="infer",
        timeout=30,
        cache_enabled=False,
    )


@pytest.fixture
def mock_response(mocker: MockerFixture) -> Any:
    """Create a mock streaming response."""
    response = mocker.Mock()
    return response
