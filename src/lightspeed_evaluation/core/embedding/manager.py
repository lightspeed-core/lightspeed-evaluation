"""Embedding Manager - Generic embedding configuration, validation, and parameter provider."""

from lightspeed_evaluation.core.models import EmbeddingConfig, SystemConfig
from lightspeed_evaluation.core.system.env_validator import validate_provider_env


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
        if self.config.provider == "openai":
            validate_provider_env("openai")
        elif self.config.provider == "huggingface":
            pass
        else:
            raise EmbeddingError(
                f"Unsupported embedding provider {self.config.provider}"
            )

    @classmethod
    def from_system_config(cls, system_config: SystemConfig) -> "EmbeddingManager":
        """Create Embedding Manager from system configuration."""
        return cls(system_config.embedding)
