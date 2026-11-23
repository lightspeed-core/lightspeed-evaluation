"""Adapters implementing the lib_side client protocols over managed clients.

Approach B hands its metrics a judge at construction. Routing that judge
through BaseCustomLLM keeps token tracking, the disk cache, SSL policy and
retry config inside the framework instead of inside the metric library.
"""

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.llm.custom import BaseCustomLLM
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.metrics.uaem.common import call_json

T = TypeVar("T", bound=BaseModel)


class UAEMJudgeLLM:  # pylint: disable=too-few-public-methods
    """Implements ``lib_side.llm.JudgeLLM`` over the managed judge client."""

    def __init__(self, llm_manager: LLMManager) -> None:
        """Build the judge client from an LLM manager."""
        self._llm = BaseCustomLLM(
            llm_manager.get_model_name(), llm_manager.get_llm_params()
        )

    def generate(
        self, *, system_prompt: str, user_prompt: str, output_model: type[T]
    ) -> T:
        """Run one structured-output judge call."""
        return call_json(
            self._llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_model=output_model,
        )


class UAEMEmbeddingClient:  # pylint: disable=too-few-public-methods
    """Implements ``lib_side.llm.EmbeddingClient`` over EmbeddingManager.

    Unused by the coverage metrics; present because the metric constructor
    contract asks for an embedding client.
    """

    def __init__(self, embedding_manager: EmbeddingManager) -> None:
        """Store the embedding manager for lazy use."""
        self._manager = embedding_manager

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed ``texts`` with the configured provider.

        Raises:
            NotImplementedError: No UAEM coverage metric needs embeddings yet.
        """
        del texts
        raise NotImplementedError(
            "No UAEM coverage metric uses embeddings; wire this up when one does."
        )
