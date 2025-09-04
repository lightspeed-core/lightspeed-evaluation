"""Lightspeed Stack Client for API communication."""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import requests

@dataclass
class LightspeedStackConfig:
    """Configuration for Lightspeed Stack API client."""

    base_url: str = "http://localhost:8080"
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    system_prompt: Optional[str] = None
    no_tools: bool = False
    timeout: int = 300


class LightspeedStackClient:
    """Client for interacting with Lightspeed Stack API."""

    def __init__(self, config: LightspeedStackConfig):
        """Initialize the client with configuration."""
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a query to the Lightspeed Stack API.

        Args:
            query: The question/query to ask
            conversation_id: Optional conversation ID for context
            attachments: Optional list of attachments

        Returns:
            Dictionary containing conversation_id and response
        """
        payload = {
            "query": query,
            "provider": self.config.provider,
            "model": self.config.model,
            "no_tools": self.config.no_tools,
        }

        if conversation_id:
            payload["conversation_id"] = conversation_id
        if self.config.system_prompt:
            payload["system_prompt"] = self.config.system_prompt
        if attachments:
            payload["attachments"] = attachments

        response = self.session.post(
            f"{self.base_url}/v1/query", json=payload, timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()

    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        response = self.session.get(f"{self.base_url}/v1/info")
        response.raise_for_status()
        return response.json()

    @classmethod
    def from_system_config(
        cls, system_config: Dict[str, Any]
    ) -> "LightspeedStackClient":
        """
        Create LightspeedStackClient from system configuration.

        Expected config structure:
        lightspeed_stack:
          base_url: "http://localhost:8080"
          provider: "openai"
          model: "gpt-4o-mini"
          system_prompt: "You are a helpful assistant..."
          no_tools: false
          timeout: 300
        """
        stack_config_dict = system_config.get("lightspeed_stack", {})

        config = LightspeedStackConfig(
            base_url=stack_config_dict.get("base_url", "http://localhost:8080"),
            provider=stack_config_dict.get("provider", "openai"),
            model=stack_config_dict.get("model", "gpt-4o-mini"),
            system_prompt=stack_config_dict.get("system_prompt"),
            no_tools=stack_config_dict.get("no_tools", False),
            timeout=stack_config_dict.get("timeout", 300),
        )

        return cls(config)
