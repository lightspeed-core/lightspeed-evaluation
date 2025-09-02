"""Lightspeed RAG LLM Manager for evaluation framework integration."""

import json
import requests
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .manager import LLMManager, LLMConfig
from ..config.models import TurnData


@dataclass
class LightspeedRAGConfig:
    """Configuration for Lightspeed RAG integration."""
    
    base_url: str = "http://localhost:8080"
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    system_prompt: Optional[str] = None
    no_tools: bool = False
    timeout: int = 300


class LightspeedRAGClient:
    """Client for interacting with Lightspeed Stack RAG API."""
    
    def __init__(self, config: LightspeedRAGConfig):
        """Initialize the client with configuration."""
        self.config = config
        self.base_url = config.base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Send a query to the RAG system.
        
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
            "no_tools": self.config.no_tools
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
        if self.config.system_prompt:
            payload["system_prompt"] = self.config.system_prompt
        if attachments:
            payload["attachments"] = attachments
            
        response = self.session.post(
            f"{self.base_url}/v1/query",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_info(self) -> Dict[str, Any]:
        """Get service information."""
        response = self.session.get(f"{self.base_url}/v1/info")
        response.raise_for_status()
        return response.json()


class LightspeedRAGManager:
    """
    LLM Manager that integrates with Lightspeed RAG for generating responses.
    
    This manager can be used within the evaluation framework to generate RAG-based
    responses that can then be evaluated using various metrics.
    """
    
    def __init__(self, rag_config: LightspeedRAGConfig):
        """Initialize with RAG configuration."""
        self.rag_config = rag_config
        self.rag_client = LightspeedRAGClient(rag_config)
        
        # Test connection
        try:
            info = self.rag_client.get_info()
            print(f"Connected to Lightspeed RAG: {info.get('name', 'Unknown')} v{info.get('version', 'Unknown')}")
        except Exception as e:
            print(f"Warning: Could not connect to Lightspeed RAG service: {e}")
    
    def generate_response(
        self,
        query: str,
        contexts: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate a response using the RAG system.
        
        Args:
            query: The user query
            contexts: Optional contexts to include as attachments
            conversation_id: Optional conversation ID for multi-turn context
            system_prompt: Optional system prompt override
            
        Returns:
            Tuple of (response, conversation_id)
        """
        # Prepare attachments from contexts if provided
        attachments = None
        if contexts:
            attachments = [
                {
                    "attachment_type": "configuration",
                    "content": context,
                    "content_type": "text/plain"
                }
                for context in contexts
            ]
        
        # Use system prompt override if provided
        effective_system_prompt = system_prompt or self.rag_config.system_prompt
        
        # Create a temporary config for this query if system prompt override is needed
        if effective_system_prompt != self.rag_config.system_prompt:
            temp_config = LightspeedRAGConfig(
                base_url=self.rag_config.base_url,
                provider=self.rag_config.provider,
                model=self.rag_config.model,
                system_prompt=effective_system_prompt,
                no_tools=self.rag_config.no_tools,
                timeout=self.rag_config.timeout
            )
            temp_client = LightspeedRAGClient(temp_config)
            result = temp_client.query(
                query=query,
                conversation_id=conversation_id,
                attachments=attachments
            )
        else:
            result = self.rag_client.query(
                query=query,
                conversation_id=conversation_id,
                attachments=attachments
            )
        
        return result.get("response", ""), result.get("conversation_id", "")
    
    def generate_response_from_turn_data(
        self,
        turn_data: TurnData,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """
        Generate a response from turn data structure.
        
        Args:
            turn_data: Turn data containing query and contexts
            conversation_id: Optional conversation ID
            system_prompt: Optional system prompt override
            
        Returns:
            Tuple of (response, conversation_id)
        """
        contexts = None
        if turn_data.contexts:
            contexts = [ctx.content for ctx in turn_data.contexts]
        
        return self.generate_response(
            query=turn_data.query,
            contexts=contexts,
            conversation_id=conversation_id,
            system_prompt=system_prompt
        )
    
    @classmethod
    def from_system_config(cls, system_config: Dict[str, Any]) -> "LightspeedRAGManager":
        """
        Create LightspeedRAGManager from system configuration.
        
        Expected config structure:
        lightspeed_rag:
          base_url: "http://localhost:8080"
          provider: "openai"
          model: "gpt-4o-mini"
          system_prompt: "You are a helpful assistant..."
          no_tools: false
          timeout: 300
        """
        rag_config_dict = system_config.get("lightspeed_rag", {})
        
        config = LightspeedRAGConfig(
            base_url=rag_config_dict.get("base_url", "http://localhost:8080"),
            provider=rag_config_dict.get("provider", "openai"),
            model=rag_config_dict.get("model", "gpt-4o-mini"),
            system_prompt=rag_config_dict.get("system_prompt"),
            no_tools=rag_config_dict.get("no_tools", False),
            timeout=rag_config_dict.get("timeout", 300)
        )
        
        return cls(config)


class LightspeedRAGLLMWrapper:
    """
    Wrapper to make LightspeedRAGManager compatible with evaluation framework expectations.
    
    This provides a bridge between the RAG manager and the evaluation metrics that
    expect standard LLM manager interfaces.
    """
    
    def __init__(self, rag_manager: LightspeedRAGManager, llm_manager: LLMManager):
        """Initialize with both RAG manager and standard LLM manager."""
        self.rag_manager = rag_manager
        self.llm_manager = llm_manager  # For compatibility with evaluation metrics
        
    def get_model_name(self) -> str:
        """Get model name for evaluation reporting."""
        return f"lightspeed_rag/{self.rag_manager.rag_config.model}"
    
    def get_litellm_params(self) -> Dict[str, Any]:
        """Get LiteLLM-compatible parameters."""
        return self.llm_manager.get_litellm_params()
    
    def get_config(self) -> LLMConfig:
        """Get LLM configuration."""
        return self.llm_manager.get_config()
    
    def generate_rag_response(
        self,
        query: str,
        contexts: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generate RAG response."""
        return self.rag_manager.generate_response(
            query=query,
            contexts=contexts,
            conversation_id=conversation_id,
            system_prompt=system_prompt
        )
    
    @classmethod
    def from_system_config(cls, system_config: Dict[str, Any]) -> "LightspeedRAGLLMWrapper":
        """Create wrapper from system configuration."""
        rag_manager = LightspeedRAGManager.from_system_config(system_config)
        llm_manager = LLMManager.from_system_config(system_config)
        return cls(rag_manager, llm_manager)