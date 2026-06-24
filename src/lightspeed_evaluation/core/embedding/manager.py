"""Embedding Manager - Generic embedding configuration, validation, and parameter provider."""

import logging

from lightspeed_evaluation.core.models import EmbeddingConfig, SystemConfig
from lightspeed_evaluation.core.system.env_validator import validate_provider_env

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Embedding config errors."""


def check_huggingface_available() -> None:
    """Check if sentence-transformers dependency is available.

    Raises:
        EmbeddingError: If required packages are not installed.
    """
    try:
        import sentence_transformers  # type: ignore[import-not-found] # noqa: F401 # pylint: disable=import-outside-toplevel,unused-import
    except ImportError as e:
        raise EmbeddingError(
            "HuggingFace embedding provider requires sentence-transformers.\n"
            "This is optional to avoid downloading large packages (torch ~6GB).\n\n"
            "Install with one of:\n"
            "  pip install 'lightspeed-evaluation[local-embeddings]'\n"
            "  uv sync --extra local-embeddings\n"
            "  pip install sentence-transformers\n\n"
            "Or use a remote embedding provider (openai, gemini) in your config:\n"
            "  embedding:\n"
            "    provider: openai\n"
            "    model: text-embedding-3-small"
        ) from e


class EmbeddingManager:
    """Generic Embedding Manager with lazy validation.

    Validation of provider environment variables is deferred until first actual
    use. This allows the evaluation pipeline to start without requiring embedding
    provider credentials when no metric relying on embeddings is configured.
    """

    def __init__(self, config: EmbeddingConfig):
        """Initialize with config. Validation is deferred until ensure_ready() is called."""
        self.config = config
        self._validated = False

    def ensure_ready(self) -> None:
        """Validate provider config and environment variables on first use.

        This method is idempotent - subsequent calls after successful validation
        are no-ops. Should be called before accessing embedding functionality.

        Raises:
            EmbeddingError: If provider is unsupported or env vars are missing.
        """
        if self._validated:
            return
        self._validate_config()
        self._validated = True
        logger.info(
            "Embedding Manager ready: %s -- %s",
            self.config.provider,
            self.config.model,
        )

    def _validate_config(self) -> None:
        """Validate config and env variables."""
        if self.config.provider == "openai":
            validate_provider_env("openai")
        elif self.config.provider == "gemini":
            validate_provider_env("gemini")
        elif self.config.provider == "huggingface":
            check_huggingface_available()
        else:
            raise EmbeddingError(
                f"Unsupported embedding provider {self.config.provider}"
            )

    @classmethod
    def from_system_config(cls, system_config: SystemConfig) -> "EmbeddingManager":
        """Create Embedding Manager from system configuration."""
        return cls(system_config.embedding)
