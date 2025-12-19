"""Ragas Embedding Manager - Ragas specific embedding wrapper."""

import logging
from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from ragas.cache import DiskCacheBackend
from ragas.embeddings import LangchainEmbeddingsWrapper

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class RagasEmbeddingManager:  # pylint: disable=too-few-public-methods
    """Ragas Embedding Manager, modifies global ragas settings."""

    def __init__(self, embedding_manager: EmbeddingManager):
        """Init RagasEmbeddingManager."""
        config = embedding_manager.config
        self.config = config

        embedding_class: Any
        if config.provider == "openai":
            embedding_class = OpenAIEmbeddings
        elif config.provider == "huggingface":
            # EmbeddingManager already validated sentence-transformers is available
            embedding_class = HuggingFaceEmbeddings
        elif config.provider == "gemini":
            embedding_class = GoogleGenerativeAIEmbeddings
        else:
            logger.error("Unknown embedding provider: %s", config.provider)
            raise ConfigurationError(f"Unknown embedding provider {config.provider}")

        logger.debug(
            "Using embedding provider: %s with model: %s",
            config.provider,
            config.model,
        )

        kwargs = config.provider_kwargs
        if kwargs is None:
            kwargs = {}

        cacher = None
        if config.cache_enabled:
            cacher = DiskCacheBackend(cache_dir=config.cache_dir)
        self.embeddings = LangchainEmbeddingsWrapper(
            embedding_class(model=config.model, **kwargs), cache=cacher
        )
