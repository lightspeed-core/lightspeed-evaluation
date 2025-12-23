"""Embedding Manager - Generic embedding configuration, validation, and parameter provider."""

from lightspeed_evaluation.core.models import EmbeddingConfig, SystemConfig
from lightspeed_evaluation.core.system.env_validator import validate_provider_env


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


class EmbeddingManager:  # pylint: disable=too-few-public-methods
    """Generic Embedding Manager."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize with validated environment and constructed model name."""
        self.config = config
        self._validate_config()
        print(
            f"""
✅ Embedding Manager: {self.config.provider} -- {self.config.model} {self.config.provider_kwargs}"""
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
