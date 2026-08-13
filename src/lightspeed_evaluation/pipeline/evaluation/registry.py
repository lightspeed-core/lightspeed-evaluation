"""Agent driver registry for evaluation pipeline."""

from __future__ import annotations

import logging
from typing import Any, Optional

from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.pipeline.evaluation.driver import (
    AgentDriver,
    AgenticRunDriver,
    HttpApiDriver,
)

logger = logging.getLogger(__name__)

AGENT_DRIVERS: dict[str, type[AgentDriver]] = {
    "http_api": HttpApiDriver,
    "agentic_run": AgenticRunDriver,
    "proposal": AgenticRunDriver,  # Deprecated: maps to AgenticRunDriver
}


class AgentDriverRegistry:  # pylint: disable=too-few-public-methods
    """Registry for creating agent drivers."""

    def __init__(self, drivers: Optional[dict[str, type[AgentDriver]]] = None) -> None:
        """Initialize the driver registry."""
        self._drivers = AGENT_DRIVERS if drivers is None else drivers

    def create_driver(
        self, agent_config: dict[str, Any], *, enabled: bool = True
    ) -> AgentDriver:
        """Create a driver instance from resolved agent configuration."""
        agent_type = agent_config.get("type")
        if not agent_type:
            raise ConfigurationError("Agent config missing required 'type' field")

        # Emit deprecation warning for "proposal" type
        if agent_type == "proposal":
            logger.warning(
                "Agent type 'proposal' is deprecated. Use 'agentic_run' instead. "
                "Support for 'proposal' will be removed in a future release."
            )

        driver_cls = self._drivers.get(agent_type)
        if not driver_cls:
            raise ConfigurationError(
                f"Unsupported agent type '{agent_type}'. "
                f"Supported types: {sorted(self._drivers)}"
            )
        return driver_cls(agent_config, enabled=enabled)
