"""Pytest configuration and fixtures for api tests."""

from typing import Any

import pytest

from pytest_mock import MockerFixture
from lightspeed_evaluation.core.models import APIConfig


@pytest.fixture
def api_config() -> APIConfig:
    """Create test API config."""
    return APIConfig(
        enabled=True,
        api_base="http://localhost:8080",
        version="v1",
        endpoint_type="query",
        timeout=30,
        cache_enabled=False,
    )


@pytest.fixture
def basic_api_config() -> APIConfig:
    """Create basic API configuration for streaming."""
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
def mock_response(mocker: MockerFixture) -> Any:
    """Create a mock streaming response."""
    response = mocker.Mock()
    return response
