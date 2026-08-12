"""Load per-run output files for consolidation."""

import csv
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
    case_results: Optional[list[dict[str, str]]] = None


def load_run_data(output_dir: str, run_index: int) -> Optional[RunData]:
    """Load summary, quality report, and case results from a run directory.

    Finds files by suffix pattern (*_summary.json, *_quality_report.json,
    *_detailed.csv).

    Args:
        output_dir: Path to the run output directory.
        run_index: Run index for this data.

    Returns:
        RunData with summary (required), quality and case_results (optional).
        None if directory missing or no usable *_summary.json found.
    """
    run_path = Path(output_dir)
    if not run_path.is_dir():
        logger.warning("Run directory not found: %s", output_dir)
        return None

    summary_dict = _load_json_by_suffix(run_path, "_summary.json")
    if summary_dict is None:
        logger.warning("No usable *_summary.json in %s", output_dir)
        return None

    quality_dict = _load_json_by_suffix(run_path, "_quality_report.json")
    case_results = _load_case_results(run_path)

    return RunData(
        run_index=run_index,
        summary=summary_dict,
        quality=quality_dict,
        case_results=case_results,
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


def _load_case_results(directory: Path) -> Optional[list[dict[str, str]]]:
    """Load per-case results from CSV for pass@k computation."""
    matches = sorted(directory.glob("*_detailed.csv"))
    if not matches:
        return None
    required = {"conversation_group_id", "turn_id", "metric_identifier", "result"}
    try:
        cases: list[dict[str, str]] = []
        with open(matches[0], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not required.issubset(set(reader.fieldnames or [])):
                logger.warning("CSV missing required columns in %s", matches[0])
                return None
            for row in reader:
                cases.append(
                    {
                        "conversation_group_id": row["conversation_group_id"],
                        "turn_id": row["turn_id"],
                        "metric_identifier": row["metric_identifier"],
                        "result": row.get("result", "ERROR"),
                    }
                )
        return cases or None
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.warning("Failed to load CSV %s: %s", matches[0], exc)
        return None
