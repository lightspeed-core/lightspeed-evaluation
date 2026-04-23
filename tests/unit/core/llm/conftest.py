"""Pytest configuration and fixtures for llm tests."""

from typing import Any, Callable

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import LLMConfig


@pytest.fixture
def llm_params() -> dict:
    """Create sample LLM parameters."""
    return {
        "timeout": 120,
        "num_retries": 5,
        "parameters": {
            "temperature": 0.5,
            "max_completion_tokens": 1024,
        },
    }


@pytest.fixture
def basic_llm_config() -> LLMConfig:
    """Create basic LLM configuration."""
    return LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.0,
        max_tokens=512,
        timeout=60,
        num_retries=3,
    )


@pytest.fixture
def mock_judge_llm_response(mocker: MockerFixture) -> Callable[..., Any]:
    """Create a mock LLM response with usage data.

    Args:
        mocker: pytest-mock fixture

    Returns:
        A factory function that creates mock response objects with configurable token counts.
        Accepts Any type for token values to support error testing with invalid values.
        If completion_tokens is None, it won't be set on the response.
    """

    def _create_response(
        prompt_tokens: Any, completion_tokens: Any, cache_hit: bool, content: str
    ) -> Any:
        mock_choice = mocker.Mock()
        mock_choice.message.content = content
        mock_response = mocker.Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage.prompt_tokens = prompt_tokens
        mock_response.usage.completion_tokens = completion_tokens

        setattr(
            mock_response,
            "_hidden_params",
            {"cache_hit": cache_hit} if cache_hit else {},
        )
        return mock_response

    return _create_response


@pytest.fixture
def mock_embedding_response(mocker: MockerFixture) -> Callable[..., Any]:
    """Create a mock embedding response with usage data.

    Args:
        mocker: pytest-mock fixture

    Returns:
        A factory function that creates mock embedding response objects.
        Accepts Any type for token values to support error testing with invalid values.
    """

    def _create_response(
        prompt_tokens: Any, cache_hit: bool, embedding: list[float] | None = None
    ) -> Any:
        mock_data_item = mocker.Mock()
        mock_data_item.object = "embedding"
        mock_data_item.embedding = embedding
        mock_data_item.index = 0

        mock_response = mocker.Mock()
        mock_response.object = "list"
        mock_response.data = [mock_data_item]
        mock_response.model = "text-embedding-3-small"

        # Create usage mock with only prompt_tokens (embeddings don't have completion_tokens)
        mock_response.usage = mocker.Mock(
            prompt_tokens=prompt_tokens, spec=["prompt_tokens"]
        )

        setattr(
            mock_response,
            "_hidden_params",
            {"cache_hit": cache_hit} if cache_hit else {},
        )
        return mock_response

    return _create_response
