"""Ragas Embedding Manager - Ragas 0.4+ specific embedding wrapper."""

import logging
from typing import Any, cast

from ragas.embeddings.base import BaseRagasEmbedding, embedding_factory

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class RagasEmbeddingManager:  # pylint: disable=too-few-public-methods
    """Ragas Embedding Manager using embedding_factory for ragas 0.4+."""

    def __init__(self, embedding_manager: EmbeddingManager):
        """Initialize RagasEmbeddingManager with embedding_factory.

        Args:
            embedding_manager: Pre-configured EmbeddingManager with validated parameters
        """
        self.config = embedding_manager.config

        # Map provider names to litellm format
        provider = self.config.provider.lower()
        model = self.config.model
        # Get additional provider kwargs
        kwargs: dict[str, Any] = {}
        if self.config.provider_kwargs:
            kwargs.update(self.config.provider_kwargs)

        # Provider-specific configuration
        if provider in ["openai", "gemini"]:
            logger.debug("Using %s provider with model: %s", provider, model)
            actual_provider = (
                "litellm"  # Litellm provider auto-creates client in embedding_factory
            )
        elif provider == "huggingface":
            # HuggingFace default is use_api=False (local sentence-transformers)
            # Only set explicitly if user hasn't overridden in provider_kwargs
            kwargs.setdefault("use_api", False)
            logger.debug(
                "Using HuggingFace provider with model: %s (local=%s)",
                model,
                not kwargs["use_api"],
            )
            actual_provider = "huggingface"
        else:
            logger.error("Unknown embedding provider: %s", self.config.provider)
            raise ConfigurationError(
                f"Unknown embedding provider {self.config.provider}"
            )

        # Create embeddings using ragas 0.4+ embedding_factory
        # Cast to BaseRagasEmbedding as embedding_factory returns union type
        self.embeddings: BaseRagasEmbedding = cast(
            BaseRagasEmbedding,
            embedding_factory(
                provider=actual_provider,
                model=model,
                **kwargs,
            ),
        )

        logger.info("Ragas Embedding Manager configured: %s/%s", provider, model)
