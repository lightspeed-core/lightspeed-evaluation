"""Ragas Embedding Manager - Ragas specific embedding wrapper."""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager


class RagasEmbeddingManager:  # pylint: disable=too-few-public-methods
    """Ragas Embedding Manager, modifies global ragas settings."""

    def __init__(self, embedding_manager: EmbeddingManager):
        """Init RagasEmbeddingManager."""
        config = embedding_manager.config
        self.config = config

        embedding_class = {
            "openai": OpenAIEmbeddings,
            "huggingface": HuggingFaceEmbeddings,
        }.get(config.provider)
        if not embedding_class:
            raise RuntimeError(f"Unknown embedding provider {config.provider}")

        kwargs = config.provider_kwargs
        if kwargs is None:
            kwargs = {}

        self.embeddings = LangchainEmbeddingsWrapper(
            embedding_class(model=config.model, **kwargs)
        )
