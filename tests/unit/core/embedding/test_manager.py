"""Unit tests for EmbeddingManager."""

import pytest
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.embedding.manager import (
    EmbeddingError,
    EmbeddingManager,
)
from lightspeed_evaluation.core.models import EmbeddingConfig
from lightspeed_evaluation.core.system.exceptions import LLMError


class TestEnsureReadyIdempotency:
    """Test that ensure_ready() is idempotent."""

    def test_second_call_is_noop(self, mocker: MockerFixture) -> None:
        """Calling ensure_ready() twice should only validate once."""
        validate_env = mocker.patch(
            "lightspeed_evaluation.core.embedding.manager.validate_provider_env"
        )
        manager = EmbeddingManager(EmbeddingConfig(provider="openai"))

        manager.ensure_ready()
        manager.ensure_ready()

        validate_env.assert_called_once()

    def test_not_validated_after_failure(self, mocker: MockerFixture) -> None:
        """Validation failure should not mark manager as ready."""
        validate_env = mocker.patch(
            "lightspeed_evaluation.core.embedding.manager.validate_provider_env",
            side_effect=[EmbeddingError("env var missing"), None],
        )
        manager = EmbeddingManager(EmbeddingConfig(provider="openai"))

        with pytest.raises(EmbeddingError):
            manager.ensure_ready()

        manager.ensure_ready()
        assert validate_env.call_count == 2


class TestEnsureReadyErrors:
    """Test ensure_ready() error cases."""

    def test_unsupported_provider_raises_embedding_error(
        self, mocker: MockerFixture
    ) -> None:
        """Unsupported provider should raise EmbeddingError.

        Pydantic validates allowed providers at construction, so we bypass
        it to exercise the defensive guard in _validate_config().
        """
        config = mocker.MagicMock()
        config.provider = "unsupported_provider"
        manager = EmbeddingManager(config)

        with pytest.raises(EmbeddingError, match="Unsupported embedding provider"):
            manager.ensure_ready()

    def test_missing_env_vars_raises(self, mocker: MockerFixture) -> None:
        """Missing env vars should propagate the error from validate_provider_env."""
        mocker.patch(
            "lightspeed_evaluation.core.embedding.manager.validate_provider_env",
            side_effect=LLMError("OPENAI_API_KEY not set"),
        )
        manager = EmbeddingManager(EmbeddingConfig(provider="openai"))

        with pytest.raises(LLMError, match="OPENAI_API_KEY not set"):
            manager.ensure_ready()

    def test_huggingface_missing_deps_raises(self, mocker: MockerFixture) -> None:
        """Missing sentence-transformers should raise EmbeddingError."""
        mocker.patch(
            "lightspeed_evaluation.core.embedding.manager.check_huggingface_available",
            side_effect=EmbeddingError("requires sentence-transformers"),
        )
        manager = EmbeddingManager(EmbeddingConfig(provider="huggingface"))

        with pytest.raises(EmbeddingError, match="sentence-transformers"):
            manager.ensure_ready()
