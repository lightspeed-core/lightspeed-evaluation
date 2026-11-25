"""Registers the ``uaem_lib`` metrics framework.

The ``lib_side`` metric library executes the metrics; this package builds a
judge client and an embedding client and injects them at construction, so
token tracking, the disk cache, SSL policy and retry config stay in this
framework rather than the library.

Requires the ``uaem-poc`` extra. The handler module imports ``lib_side``
lazily, so a plain install is unaffected and the framework simply does not
register.
"""

from lightspeed_evaluation.core.metrics.uaem.common import (
    DEFAULT_RAW_SCORE_PATH,
    FRAMEWORK,
    UAEM_METRICS,
    RawScoreSink,
    ScoredMetric,
    build_traces,
    call_json,
    resolve_raw_score_path,
    to_lse_score,
)
from lightspeed_evaluation.core.metrics.uaem.judge import (
    UAEMEmbeddingClient,
    UAEMJudgeLLM,
)
from lightspeed_evaluation.core.metrics.uaem.metrics import UAEMLibMetrics
from lightspeed_evaluation.core.metrics.uaem.rollup import build_rollup_summary

__all__ = [
    "DEFAULT_RAW_SCORE_PATH",
    "FRAMEWORK",
    "UAEM_METRICS",
    "RawScoreSink",
    "ScoredMetric",
    "UAEMEmbeddingClient",
    "UAEMJudgeLLM",
    "UAEMLibMetrics",
    "build_rollup_summary",
    "build_traces",
    "call_json",
    "resolve_raw_score_path",
    "to_lse_score",
]
