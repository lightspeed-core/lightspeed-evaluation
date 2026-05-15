"""Unit tests for RagasEmbeddingManager."""

import pytest
from pytest_mock import MockerFixture, MockType

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.embedding.ragas import RagasEmbeddingManager
from lightspeed_evaluation.core.models import EmbeddingConfig


class TestRagasEmbeddingManagerProviderRouting:
    """Test provider routing logic in RagasEmbeddingManager."""

    def test_huggingface_uses_native_provider(
        self, mocker: MockerFixture, mock_embedding_factory: MockType
    ) -> None:
        """Verify HuggingFace uses native provider, not litellm.

        HuggingFace embeddings should use the 'huggingface' provider directly
        to support local sentence-transformers models with use_api=False.
        """
        mocker.patch(
            "lightspeed_evaluation.core.embedding.manager.check_huggingface_available"
        )

        config = EmbeddingConfig(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        embedding_manager = EmbeddingManager(config)

        RagasEmbeddingManager(embedding_manager)

        mock_embedding_factory.assert_called_once_with(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            use_api=False,
        )

    @pytest.mark.parametrize(
        "provider,model",
        [
            ("openai", "text-embedding-3-small"),
            ("gemini", "text-embedding-004"),
        ],
    )
    def test_cloud_providers_use_litellm(
        self,
        mocker: MockerFixture,
        mock_embedding_factory: MockType,
        provider: str,
        model: str,
    ) -> None:
        """Verify OpenAI and Gemini use litellm provider.

        Cloud providers (OpenAI, Gemini) should route through 'litellm'
        which auto-creates the appropriate client in embedding_factory.
        """
        mocker.patch(
            "lightspeed_evaluation.core.embedding.manager.validate_provider_env"
        )

        config = EmbeddingConfig(provider=provider, model=model)
        embedding_manager = EmbeddingManager(config)

        RagasEmbeddingManager(embedding_manager)

        mock_embedding_factory.assert_called_once_with(
            provider="litellm",
            model=model,
        )
