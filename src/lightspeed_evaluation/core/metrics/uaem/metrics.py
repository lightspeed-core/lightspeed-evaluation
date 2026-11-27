"""UAEM metrics handler.

The metric library (``lib_side``) owns execution; this handler builds the
judge and embedding clients and hands them to the metric constructors at
construction time.
"""

import logging
from typing import Any, Optional

from pydantic import ValidationError

from lightspeed_evaluation.core.embedding.manager import EmbeddingManager
from lightspeed_evaluation.core.llm.manager import LLMManager
from lightspeed_evaluation.core.metrics.uaem.common import (
    FRAMEWORK,
    build_traces,
    format_pass_condition,
)
from lightspeed_evaluation.core.metrics.uaem.judge import (
    UAEMEmbeddingClient,
    UAEMJudgeLLM,
)
from lightspeed_evaluation.core.models import EvaluationScope, ScoredMetric
from lightspeed_evaluation.core.system.exceptions import LLMError

logger = logging.getLogger(__name__)


class UAEMLibMetrics:  # pylint: disable=too-few-public-methods
    """UAEM metrics handler, bound to one judge at construction."""

    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_manager: Optional[EmbeddingManager] = None,
    ) -> None:
        """Construct every UAEM metric with injected clients.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters.
            embedding_manager: Embedding manager; unused by these metrics.

        Raises:
            ImportError: If the uaem-poc extra is not installed.
        """
        # Imported lazily: lib_side ships with the uaem-poc extra only.
        from lib_side import build_metrics  # pylint: disable=import-outside-toplevel

        self.metrics = build_metrics(
            llm=UAEMJudgeLLM(llm_manager),
            embeddings=(
                UAEMEmbeddingClient(embedding_manager)
                if embedding_manager is not None
                else None
            ),
        )

        logger.info("UAEM lib-side metrics initialized: %s", sorted(self.metrics))

    def evaluate(
        self, metric_name: str, conv_data: Any, scope: EvaluationScope
    ) -> ScoredMetric:
        """Evaluate one self-executing UAEM metric.

        Args:
            metric_name: Metric name within the uaem_lib framework.
            conv_data: Conversation data for the evaluation.
            scope: Turn or conversation context.

        Returns:
            The result on both scales; ``score`` is None when the metric
            cannot run.
        """
        metric = self.metrics.get(metric_name)
        if metric is None:
            return ScoredMetric(reason=f"Unsupported {FRAMEWORK} metric: {metric_name}")

        traces = build_traces(conv_data, scope)
        if not traces:
            return ScoredMetric(reason="No usable query/response pairs for this metric")

        try:
            # The library states its own result on both scales, and owns
            # the native -> [0, 1] transform: the rollup applies the same
            # shape at the indicator level, so a second copy of it here
            # would land it twice.
            scored = metric.score(traces=traces)
        except (LLMError, ValidationError, ValueError) as exc:
            return ScoredMetric(reason=f"{metric_name} failed: {exc}")

        if scored.score is None or scored.raw_score is None:
            return ScoredMetric(
                reason=scored.reason or f"{metric_name} returned a null score"
            )

        return ScoredMetric(
            score=scored.score,
            reason=(
                f"{metric_name}={scored.raw_score:.4f}; {scored.reason} "
                f"(reported as {scored.score:.4f} on the [0, 1] scale)"
            ),
            raw_score=scored.raw_score,
            score_min=scored.score_min,
            score_max=scored.score_max,
            pass_condition=format_pass_condition(scored.pass_condition),
        )
