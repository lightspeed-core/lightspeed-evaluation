"""Embedding Manager - Generic embedding configuration, validation, and parameter provider."""

from ..llm import validate_openai_env
from ..models import EmbeddingConfig, SystemConfig


class EmbeddingError(Exception):
    """Embedding config errors."""


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

        def empty_check() -> None:
            pass

        env_validator = {
            "openai": validate_openai_env,
            # "google": _validate_gemini_env, # Google embeddings are not supported at the moment
            "huggingface": empty_check,
        }.get(self.config.provider)

        if env_validator is None:
            raise EmbeddingError(
                f"Unsupported embedding provider {self.config.provider}"
            )

        env_validator()

    @classmethod
    def from_system_config(cls, system_config: SystemConfig) -> "EmbeddingManager":
        """Create Embedding Manager from system configuration."""
        return cls(system_config.embedding)
