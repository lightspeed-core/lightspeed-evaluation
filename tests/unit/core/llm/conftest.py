"""Pytest configuration and fixtures for llm tests."""

import pytest

from lightspeed_evaluation.core.models import LLMConfig


@pytest.fixture
def llm_params() -> dict:
    """Create sample LLM parameters."""
    return {
        "temperature": 0.5,
        "max_completion_tokens": 1024,
        "timeout": 120,
        "num_retries": 5,
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
