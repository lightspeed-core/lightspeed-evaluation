"""Post-run UAEM rollup, shaped for the evaluation summary.

``lib_side.rollup_report`` owns the metric -> indicator -> evaluation
criterion math (see ``to_lse_score`` for the raw-vs-reported score split
this relies on). This module locates this run's sink file and extracts the
``uaem_lib`` entry from what the library returns.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from lightspeed_evaluation.core.metrics.uaem.common import (
    FRAMEWORK,
    resolve_raw_score_path,
)

logger = logging.getLogger(__name__)


def build_rollup_summary(path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """Roll this run's ``uaem_lib`` raw scores up to indicator/criterion/domain.

    Args:
        path: Raw-score sink to read; defaults to this run's resolved path.

    Returns:
        ``lib_side.rollup_report``'s tree for ``uaem_lib``, dumped to a
        JSON-safe dict, or ``None`` when the ``uaem_lib`` framework did not
        run this run (no sink file, no matching rows) or the ``uaem-poc``
        extra is not installed.
    """
    sink_path = path if path is not None else resolve_raw_score_path()
    if not sink_path.exists():
        return None

    try:
        # Imported lazily: lib_side ships with the uaem-poc extra only.
        from lib_side import rollup_report  # pylint: disable=import-outside-toplevel
    except ImportError:
        logger.debug(
            "uaem_lib raw scores present but lib_side is not installed; "
            "skipping the rollup section"
        )
        return None

    rolled = rollup_report(sink_path, model_nm=FRAMEWORK).get(FRAMEWORK)
    if rolled is None:
        return None
    return rolled.model_dump(mode="json")
