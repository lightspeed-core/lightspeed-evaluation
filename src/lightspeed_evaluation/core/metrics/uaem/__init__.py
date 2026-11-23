"""UAEM metrics, Ragas-style architecture (UIEINTEL-698, approach B).

Registers the ``uaem_lib`` framework. The metric library executes itself;
this package builds a judge client and an embedding client and injects them
at construction, so token tracking, the disk cache, SSL policy and retry
config still stay in the framework.

Requires the ``uaem-poc`` extra, which installs the metric library. The
handler module imports it lazily, so a plain install is unaffected and the
framework simply does not register.
"""

from lightspeed_evaluation.core.metrics.uaem.common import (
    DEFAULT_RAW_SCORE_PATH,
    FRAMEWORK,
    UAEM_METRICS,
    RawScoreSink,
    ScoredMetric,
    build_traces,
    call_json,
    to_lse_score,
)
from lightspeed_evaluation.core.metrics.uaem.judge import (
    UAEMEmbeddingClient,
    UAEMJudgeLLM,
)
from lightspeed_evaluation.core.metrics.uaem.metrics import UAEMLibMetrics

__all__ = [
    "DEFAULT_RAW_SCORE_PATH",
    "FRAMEWORK",
    "UAEM_METRICS",
    "RawScoreSink",
    "ScoredMetric",
    "ScoredMetric",
    "UAEMEmbeddingClient",
    "UAEMJudgeLLM",
    "UAEMLibMetrics",
    "build_traces",
    "call_json",
    "to_lse_score",
]
