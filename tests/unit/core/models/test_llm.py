"""Tests for LLM configuration models."""

import pytest
from pydantic import ValidationError

from lightspeed_evaluation.core.models.llm import (
    LLMPoolConfig,
    LLMProviderConfig,
)


class TestLLMProviderConfigDescription:
    """Tests for the description field on LLMProviderConfig."""

    def test_description_defaults_to_none(self) -> None:
        """Test description defaults to None when not provided."""
        config = LLMProviderConfig(provider="openai")
        assert config.description is None

    def test_description_field(self) -> None:
        """Test description field is accepted and stored."""
        config = LLMProviderConfig(
            provider="openai",
            model="gpt-4o-mini",
            description="Cost-efficient judge for fast evaluations",
        )
        assert config.description == "Cost-efficient judge for fast evaluations"

    def test_description_in_model_dump(self) -> None:
        """Test description is present in model_dump output."""
        config = LLMProviderConfig(
            provider="openai",
            description="My judge model",
        )
        dumped = config.model_dump()
        assert dumped["description"] == "My judge model"

    def test_description_none_in_model_dump(self) -> None:
        """Test description None is present in model_dump output."""
        config = LLMProviderConfig(provider="openai")
        dumped = config.model_dump()
        assert dumped["description"] is None


class TestLLMPoolConfigDescription:
    """Tests for description field in LLMPoolConfig context."""

    def test_pool_model_with_description(self) -> None:
        """Test description is preserved in pool model definitions."""
        pool = LLMPoolConfig.model_validate(
            {
                "models": {
                    "judge_1": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "description": "Primary judge for correctness",
                    },
                    "judge_2": {
                        "provider": "openai",
                        "model": "gpt-4.1-mini",
                    },
                }
            }
        )
        assert pool.models["judge_1"].description == "Primary judge for correctness"
        assert pool.models["judge_2"].description is None

    def test_resolve_llm_config_ignores_description(self) -> None:
        """Test that resolve_llm_config produces LLMConfig without description.

        Description is a pool-level annotation, not an operational parameter.
        LLMConfig does not have a description field.
        """
        pool = LLMPoolConfig.model_validate(
            {
                "models": {
                    "judge_1": {
                        "provider": "openai",
                        "model": "gpt-4o-mini",
                        "description": "My judge",
                    },
                }
            }
        )
        llm_config = pool.resolve_llm_config("judge_1")
        assert llm_config.provider == "openai"
        assert llm_config.model == "gpt-4o-mini"
        # LLMConfig doesn't have description - it's a pool-level annotation
        assert "description" not in type(llm_config).model_fields

    def test_description_rejected_as_extra_on_pool(self) -> None:
        """Test that description on LLMPoolConfig itself is rejected (extra=forbid)."""
        with pytest.raises(ValidationError):
            LLMPoolConfig.model_validate(
                {
                    "description": "This should fail",
                    "models": {},
                }
            )
