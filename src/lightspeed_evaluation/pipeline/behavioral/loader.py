"""Load per-run output files for consolidation."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunData:
    """Data loaded from a single run's output directory.

    Decouples consolidation from file format — future backends
    (database, remote storage) populate the same structure.
    """

    run_index: int
    summary: dict[str, Any] = field(default_factory=dict)
    quality: Optional[dict[str, Any]] = None


def load_run_data(output_dir: str, run_index: int) -> Optional[RunData]:
    """Load summary and quality report from a run's output directory.

    Args:
        output_dir: Path to the run output directory.
        run_index: Run index for this data.

    Returns:
        RunData if summary.json was found, None otherwise.
    """
    run_path = Path(output_dir)
    if not run_path.is_dir():
        logger.warning("Run directory not found: %s", output_dir)
        return None

    summary_dict = _load_json_by_suffix(run_path, "_summary.json")
    if summary_dict is None:
        logger.warning("No summary.json found in %s", output_dir)
        return None

    quality_dict = _load_json_by_suffix(run_path, "_quality_report.json")

    return RunData(
        run_index=run_index,
        summary=summary_dict,
        quality=quality_dict,
    )


def _load_json_by_suffix(directory: Path, suffix: str) -> Optional[dict[str, Any]]:
    """Find and load the first JSON file matching a suffix pattern."""
    matches = sorted(directory.glob(f"*{suffix}"))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "Multiple files matching *%s in %s, using %s", suffix, directory, matches[0]
        )
    try:
        with open(matches[0], encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            logger.warning(
                "Expected dict in %s, got %s", matches[0], type(data).__name__
            )
            return None
        return data
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        logger.warning("Failed to load %s: %s", matches[0], exc)
        return None
