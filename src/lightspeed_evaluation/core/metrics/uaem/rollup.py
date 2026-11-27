"""Post-run UAEM rollup, built from this run's results in memory.

``lib_side.rollup_scores`` owns the metric -> indicator -> evaluation
criterion -> domain math. This module only decides which scores to hand it:
the native-scale scores this run produced, grouped by the agent under
evaluation and then by evaluation level.

Grouping by agent, not by framework: the tree describes the system that was
scored, so two agents in one run get two trees. Turn- and conversation-level
scores stay in separate trees because they are means over different
populations. Runs are separate by construction — the input is one run's
results, and nothing is read back from disk.
"""

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Any, Optional

from lightspeed_evaluation.core.metrics.uaem.common import FRAMEWORK

logger = logging.getLogger(__name__)

#: Grouping key for results whose agent the framework was never told, as in
#: an offline run over pre-populated responses.
UNKNOWN_AGENT = "unknown"


def _group_native_scores(
    results: Sequence[Any],
) -> dict[str, dict[str, dict[str, list[float]]]]:
    """Collect this run's uaem_lib native scores: agent -> level -> metric.

    Reads ``raw_score``, never ``score``: the reported score already has the
    indicator's own ``1 - |x|`` shape applied for a zero-centred metric, and
    the rollup applies it again.
    """
    grouped: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for result in results:
        framework, _, metric_name = result.metric_identifier.partition(":")
        if framework != FRAMEWORK or result.raw_score is None:
            continue
        level = "conversation" if result.turn_id is None else "turn"
        grouped[result.agent or UNKNOWN_AGENT][level][metric_name].append(
            float(result.raw_score)
        )
    return grouped


def _mean_per_metric(by_metric: dict[str, list[float]]) -> dict[str, float | None]:
    """Mean each metric across rows.

    Mean first, formula second: the indicator formulas are non-linear, so
    the two orders disagree. Granularity rows of -1 and +1 mean to 0, which
    the indicator reads as "right on average" (1.0); transforming first and
    then averaging says the opposite (0.0).
    """
    return {
        name: (sum(scores) / len(scores) if scores else None)
        for name, scores in sorted(by_metric.items())
    }


def build_rollup_summary(
    results: Sequence[Any],
) -> Optional[dict[str, dict[str, Any]]]:
    """Roll this run's ``uaem_lib`` native scores up, per agent and level.

    Args:
        results: This run's evaluation results. Results from other
            frameworks, and uaem_lib results with no native score, are
            skipped.

    Returns:
        agent -> evaluation level -> the metric/indicator/evaluation
        criterion/domain tree, or ``None`` when the ``uaem_lib`` framework
        produced no native score this run or the ``uaem-poc`` extra is not
        installed.
    """
    grouped = _group_native_scores(results)
    if not grouped:
        return None

    try:
        # Imported lazily: lib_side ships with the uaem-poc extra only.
        from lib_side import rollup_scores  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.debug(
            "uaem_lib scores present but lib_side is not installed; "
            "skipping the rollup section"
        )
        return None

    return {
        agent: {
            level: rollup_scores(_mean_per_metric(by_metric)).model_dump(mode="json")
            for level, by_metric in sorted(by_level.items())
        }
        for agent, by_level in sorted(grouped.items())
    }
