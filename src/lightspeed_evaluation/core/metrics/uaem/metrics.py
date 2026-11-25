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
    RawScoreSink,
    ScoredMetric,
    build_traces,
    to_lse_score,
)
from lightspeed_evaluation.core.metrics.uaem.judge import (
    UAEMEmbeddingClient,
    UAEMJudgeLLM,
)
from lightspeed_evaluation.core.models import EvaluationScope
from lightspeed_evaluation.core.system.exceptions import LLMError

logger = logging.getLogger(__name__)


class UAEMLibMetrics:  # pylint: disable=too-few-public-methods
    """UAEM metrics handler, bound to one judge at construction."""

    def __init__(
        self,
        llm_manager: LLMManager,
        embedding_manager: Optional[EmbeddingManager] = None,
        raw_sink: Optional[RawScoreSink] = None,
    ) -> None:
        """Construct every UAEM metric with injected clients.

        Args:
            llm_manager: Pre-configured LLMManager with validated parameters.
            embedding_manager: Embedding manager; unused by these metrics.
            raw_sink: Records untransformed scores for the post-run rollup.

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
        self.raw_sink = raw_sink

        logger.info("UAEM lib-side metrics initialized: %s", sorted(self.metrics))

    def evaluate(
        self, metric_name: str, conv_data: Any, scope: EvaluationScope
    ) -> tuple[Optional[float], str]:
        """Evaluate one self-executing UAEM metric.

        Args:
            metric_name: Metric name within the uaem_lib framework.
            conv_data: Conversation data for the evaluation.
            scope: Turn or conversation context.

        Returns:
            A (score, reason) pair; score is None when the metric cannot run.
        """
        metric = self.metrics.get(metric_name)
        if metric is None:
            return None, f"Unsupported {FRAMEWORK} metric: {metric_name}"

        traces = build_traces(conv_data, scope)
        if not traces:
            return None, "No usable query/response pairs for this metric"

        try:
            result = metric.score(traces=traces)
        except (LLMError, ValidationError, ValueError) as exc:
            return None, f"{metric_name} failed: {exc}"

        if result.score is None:
            return None, result.explanation or f"{metric_name} returned a null score"

        raw_score = float(result.score)
        score = to_lse_score(
            raw_score,
            score_min=metric.metadata.score_min,
            score_max=metric.metadata.score_max,
        )
        if self.raw_sink is not None:
            self.raw_sink.record(
                ScoredMetric(metric_name, raw_score, score), conv_data, scope
            )
        return score, (
            f"{metric_name}={raw_score:.4f}; {result.explanation} "
            f"(reported as {score:.4f} on the [0, 1] scale)"
        )
