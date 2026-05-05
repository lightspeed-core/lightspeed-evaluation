"""Agent driver architecture for evaluation pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any, Optional

from lightspeed_evaluation.core.api import APIClient
from lightspeed_evaluation.core.models import HttpApiAgentConfig, TurnData
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender

logger = logging.getLogger(__name__)


class AgentDriver(ABC):
    """Abstract driver interface for agent execution."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the driver with validated config."""
        self.validate_config(config)

    @abstractmethod
    def execute_turn(
        self, turn_data: TurnData, conversation_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute a single turn and amend data in place.

        Returns:
            Tuple of (error_message, updated_conversation_id).
        """

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate agent configuration."""

    def close(self) -> None:
        """Release any resources held by the driver."""

    @property
    def enabled(self) -> bool:
        """Whether the driver should execute."""
        return True


class HttpApiDriver(AgentDriver):
    """Driver that enriches turn data via the HTTP API."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the HTTP API driver with validated config."""
        super().__init__(config)
        self._config = self._parse_config(config)
        self._enabled = self._config.enabled
        self._api_client = self._create_api_client(self._config)
        self._amender = APIDataAmender(self._api_client)

    def execute_turn(
        self, turn_data: TurnData, conversation_id: Optional[str] = None
    ) -> tuple[Optional[str], Optional[str]]:
        """Execute the HTTP API driver for a single turn.

        Returns:
            Tuple of (error_message, updated_conversation_id).
        """
        return self._amender.amend_single_turn(turn_data, conversation_id)

    def validate_config(self, config: dict[str, Any]) -> None:
        """Validate HTTP API driver configuration."""
        HttpApiAgentConfig.model_validate(config)

    def close(self) -> None:
        """Close the underlying API client."""
        if self._api_client:
            self._api_client.close()

    @property
    def enabled(self) -> bool:
        """Return whether the driver is enabled."""
        return self._enabled

    def _parse_config(self, config: dict[str, Any]) -> HttpApiAgentConfig:
        return HttpApiAgentConfig.model_validate(config)

    def _create_api_client(self, config: HttpApiAgentConfig) -> Optional[APIClient]:
        if not config.enabled:
            return None
        from lightspeed_evaluation.core.models import APIConfig

        api_config = APIConfig.model_validate(config.model_dump(exclude={"type"}))
        return APIClient(api_config)


AGENT_DRIVERS: dict[str, type[AgentDriver]] = {
    "http_api": HttpApiDriver,
}


class AgentDriverRegistry:
    """Registry for creating agent drivers."""

    def __init__(self, drivers: Optional[dict[str, type[AgentDriver]]] = None) -> None:
        """Initialize the driver registry."""
        self._drivers = drivers or AGENT_DRIVERS

    def create_driver(self, agent_config: dict[str, Any]) -> AgentDriver:
        """Create a driver instance from resolved agent configuration."""
        agent_type = agent_config.get("type")
        if not agent_type:
            raise ConfigurationError("Agent config missing required 'type' field")

        driver_cls = self._drivers.get(agent_type)
        if not driver_cls:
            raise ConfigurationError(
                f"Unsupported agent type '{agent_type}'. "
                f"Supported types: {sorted(self._drivers)}"
            )
        return driver_cls(agent_config)
