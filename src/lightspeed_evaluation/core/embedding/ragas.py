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
        config = embedding_manager.config
        self.config = config

        # Map provider names to litellm format
        provider = config.provider.lower()
        model = config.model

        # Build the model string for litellm
        # Only OpenAI, Gemini, and HuggingFace are supported
        if provider == "openai":
            model_str = model  # OpenAI models don't need prefix
        elif provider == "huggingface":
            model_str = f"huggingface/{model}"
        elif provider == "gemini":
            model_str = f"gemini/{model}"
        else:
            logger.error("Unknown embedding provider: %s", config.provider)
            raise ConfigurationError(f"Unknown embedding provider {config.provider}")

        logger.debug(
            "Using embedding provider: %s with model: %s -> %s",
            provider,
            model,
            model_str,
        )

        # Get additional provider kwargs
        kwargs: dict[str, Any] = {}
        if config.provider_kwargs:
            kwargs.update(config.provider_kwargs)

        # Create embeddings using ragas 0.4+ embedding_factory with litellm
        # Cast to BaseRagasEmbedding as embedding_factory returns union type
        self.embeddings: BaseRagasEmbedding = cast(
            BaseRagasEmbedding,
            embedding_factory(
                "litellm",
                model=model_str,
                **kwargs,
            ),
        )

        logger.info("Ragas Embedding Manager configured: %s/%s", provider, model)
