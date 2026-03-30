"""LLM Manager - Generic LLM configuration, validation, and parameter provider."""

import logging
import os
from typing import Any, Optional

from lightspeed_evaluation.core.models import LLMConfig, SystemConfig
from lightspeed_evaluation.core.system.env_validator import validate_provider_env
from lightspeed_evaluation.core.system.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class LLMManager:
    """Generic LLM Manager for all use cases (Ragas, DeepEval, Custom metrics).

    Responsibilities:
    - Environment validation for multiple providers
    - Model name construction
    - Provides LLM parameters for consumption by framework-specific managers
    - Manages judge panel configurations when available
    """

    def __init__(
        self,
        config: LLMConfig,
        system_config: Optional[SystemConfig] = None,
        judge_id: Optional[str] = None,
    ):
        """Initialize with validated environment and constructed model name.

        Args:
            config: Primary LLM configuration (also used as fallback)
            system_config: Optional full system config for judge panel support
            judge_id: Optional identifier for this judge (pool key). If not provided,
                defaults to "primary" for single LLM or the pool key for panel judges.
        """
        self.config = config
        self.system_config = system_config
        self.model_name = self._construct_model_name_and_validate(config)
        self.judge_id = judge_id or "primary"

        # Initialize judge panel if available
        self.judge_managers: list["LLMManager"] = []
        if system_config and system_config.judge_panel and system_config.llm_pool:
            panel = system_config.judge_panel
            logger.info("Judge panel configured with %d judges", len(panel.judges))
            # Create LLM managers for each judge using resolved configs from llms pool
            try:
                judge_configs = system_config.get_judge_configs()
                for pool_key, resolved_config in judge_configs:
                    # Create child manager without system_config to avoid recursion
                    judge_manager = LLMManager(resolved_config, judge_id=pool_key)
                    self.judge_managers.append(judge_manager)
            except ConfigurationError as e:
                logger.error("Failed to resolve judge panel: %s", e)
                raise
        else:
            # No judge panel - log single LLM info
            logger.info(
                "LLM Manager: %s/%s -> %s",
                self.config.provider,
                self.config.model,
                self.model_name,
            )

    def _construct_model_name_and_validate(self, config: LLMConfig) -> str:
        """Construct model name and validate required environment variables.

        Args:
            config: LLM configuration to construct model name for

        Returns:
            Constructed model name string
        """
        provider = config.provider.lower()

        # Provider-specific validation and model name construction
        provider_handlers = {
            "hosted_vllm": self._handle_hosted_vllm_provider,
            "openai": self._handle_openai_provider,
            "azure": self._handle_azure_provider,
            "watsonx": self._handle_watsonx_provider,
            "anthropic": self._handle_anthropic_provider,
            "gemini": self._handle_gemini_provider,
            "vertex": self._handle_vertex_provider,
            "ollama": self._handle_ollama_provider,
        }

        if provider in provider_handlers:
            return provider_handlers[provider]()

        # Generic provider - try as-is with warning
        logger.warning("Using generic provider format for %s", provider)
        return f"{provider}/{config.model}"

    def _handle_hosted_vllm_provider(self) -> str:
        """Handle hosted vLLM provider setup."""
        validate_provider_env("hosted_vllm")
        return f"hosted_vllm/{self.config.model}"

    def _handle_openai_provider(self) -> str:
        """Handle OpenAI provider setup."""
        validate_provider_env("openai")
        return self.config.model

    def _handle_azure_provider(self) -> str:
        """Handle Azure provider setup."""
        validate_provider_env("azure")
        deployment = os.environ.get("AZURE_DEPLOYMENT_NAME") or self.config.model
        return f"azure/{deployment}"

    def _handle_watsonx_provider(self) -> str:
        """Handle WatsonX provider setup."""
        validate_provider_env("watsonx")
        return f"watsonx/{self.config.model}"

    def _handle_anthropic_provider(self) -> str:
        """Handle Anthropic provider setup."""
        validate_provider_env("anthropic")
        return f"anthropic/{self.config.model}"

    def _handle_gemini_provider(self) -> str:
        """Handle Gemini provider setup."""
        validate_provider_env("gemini")
        return f"gemini/{self.config.model}"

    def _handle_vertex_provider(self) -> str:
        """Handle Vertex AI provider setup."""
        validate_provider_env("vertex")
        return self.config.model

    def _handle_ollama_provider(self) -> str:
        """Handle Ollama provider setup."""
        validate_provider_env("ollama")
        return f"ollama/{self.config.model}"

    def has_judge_panel(self) -> bool:
        """Check if judge panel is configured.

        Returns:
            True if judge panel is configured (one or more judges)
        """
        return len(self.judge_managers) > 0

    def get_judge_managers(self) -> list["LLMManager"]:
        """Get list of judge LLM managers.

        Returns:
            List of LLMManager instances. If no panel configured, returns list
            with single manager (self). Always returns at least one manager.
        """
        if self.judge_managers:
            return self.judge_managers
        # No panel - return self as single judge
        return [self]

    def get_primary_judge(self) -> "LLMManager":
        """Get primary judge LLM manager (first in panel or self).

        This is used when panel is disabled for specific metrics or
        as fallback when panel is not configured.

        Returns:
            Primary LLM manager (first judge if panel exists, otherwise self)
        """
        if self.judge_managers:
            return self.judge_managers[0]
        return self

    def get_judges_for_metric(self, metric_identifier: str) -> list["LLMManager"]:
        """Get list of judges to use for a specific metric.

        Returns all judges if metric should use panel, otherwise returns
        list with single primary judge. Always returns a list.

        Args:
            metric_identifier: Metric identifier (e.g., "ragas:faithfulness")

        Returns:
            List of LLMManager instances to use for this metric
        """
        if self.should_use_panel_for_metric(metric_identifier):
            return self.get_judge_managers()
        return [self.get_primary_judge()]

    def should_use_panel_for_metric(self, metric_identifier: str) -> bool:
        """Determine if a metric should use judge panel based on enabled_metrics.

        Args:
            metric_identifier: Metric identifier (e.g., "ragas:faithfulness")

        Returns:
            True if metric should use judge panel, False otherwise
        """
        if self.system_config and self.system_config.judge_panel:
            enabled_metrics = self.system_config.judge_panel.enabled_metrics
            # If enabled_metrics is None, all metrics use panel
            if enabled_metrics is None:
                return True
            # Check if this specific metric is in the list
            return metric_identifier in enabled_metrics

        # Default: Don't use panel (use primary judge only)
        return False

    def get_model_name(self) -> str:
        """Get the constructed model name."""
        return self.model_name

    def get_llm_params(self) -> dict[str, Any]:
        """Get all LLM parameters for adapter consumption.

        Returns operational params and inference parameters dict.
        Adapters read operational fields directly and spread **parameters
        for inference params (temperature, max_completion_tokens, etc.).
        """
        return {
            "model": self.model_name,
            "timeout": self.config.timeout,
            "num_retries": self.config.num_retries,
            "ssl_verify": self.config.ssl_verify,
            "parameters": dict(self.config.parameters),
        }

    def get_config(self) -> LLMConfig:
        """Get the LLM configuration."""
        return self.config

    @classmethod
    def from_system_config(cls, system_config: SystemConfig) -> "LLMManager":
        """Create LLM Manager from system configuration.

        Args:
            system_config: System configuration with LLM and optional judge panel

        Returns:
            LLMManager with judge panel support if configured
        """
        return cls(system_config.llm, system_config=system_config)

    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig) -> "LLMManager":
        """Create LLM Manager from LLMConfig directly (no judge panel support).

        Args:
            llm_config: LLM configuration

        Returns:
            LLMManager without judge panel support
        """
        return cls(llm_config)
