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
    turn_stats: Optional[list[dict[str, float]]] = None


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
    turn_stats = _load_turn_stats(run_path)

    return RunData(
        run_index=run_index,
        summary=summary_dict,
        quality=quality_dict,
        case_results=case_results,
        turn_stats=turn_stats,
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


_AGENT_FIELDS = {
    "api_input_tokens": "agent_input_tokens",
    "api_output_tokens": "agent_output_tokens",
}

_JUDGE_FIELDS = {
    "judge_llm_input_tokens": "judge_input_tokens",
    "judge_llm_output_tokens": "judge_output_tokens",
}


def _load_turn_stats(directory: Path) -> Optional[list[dict[str, float]]]:
    """Extract per-turn numeric fields from detailed CSV for percentile computation.

    Agent token fields are deduplicated by (conversation_group_id, turn_id)
    because the CSV has one row per metric — agent tokens repeat across metrics
    for the same turn. Rows without a turn_id (conversation-level) are skipped
    for agent fields since their values are conversation totals.

    Judge token fields are kept per row since each metric evaluation has its
    own judge cost.
    """
    matches = sorted(directory.glob("*_detailed.csv"))
    if not matches:
        return None
    try:
        with open(matches[0], encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            agent_cols = _available_cols(_AGENT_FIELDS, columns)
            judge_cols = _available_cols(_JUDGE_FIELDS, columns)
            if not agent_cols and not judge_cols:
                return None
            stats, seen_turns = _parse_rows(reader, agent_cols, judge_cols)
            stats.extend(seen_turns.values())
        return stats or None
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        logger.warning("Failed to load turn stats from %s: %s", matches[0], exc)
        return None


def _available_cols(
    field_map: dict[str, str], columns: set[str]
) -> list[tuple[str, str]]:
    """Return (csv_col, out_key) pairs for columns present in the CSV."""
    return [(c, k) for c, k in field_map.items() if c in columns]


def _parse_rows(
    reader: csv.DictReader,  # type: ignore[type-arg]
    agent_cols: list[tuple[str, str]],
    judge_cols: list[tuple[str, str]],
) -> tuple[list[dict[str, float]], dict[tuple[str, str], dict[str, float]]]:
    """Parse CSV rows into judge stats list and deduplicated agent map."""
    stats: list[dict[str, float]] = []
    seen_turns: dict[tuple[str, str], dict[str, float]] = {}
    for row in reader:
        _collect_agent_row(row, agent_cols, seen_turns)
        judge_vals = _extract_fields(row, judge_cols)
        if judge_vals:
            stats.append(judge_vals)
    return stats, seen_turns


def _collect_agent_row(
    row: dict[str, str],
    agent_cols: list[tuple[str, str]],
    seen_turns: dict[tuple[str, str], dict[str, float]],
) -> None:
    """Deduplicate agent tokens by (conv_id, turn_id), first nonzero wins."""
    if not agent_cols:
        return
    turn_id = row.get("turn_id", "")
    if not turn_id:
        return
    conv_id = row.get("conversation_group_id", "")
    turn_key = (conv_id, turn_id)
    if turn_key not in seen_turns:
        agent_vals = _extract_fields(row, agent_cols)
        if agent_vals:
            seen_turns[turn_key] = agent_vals
    else:
        existing = seen_turns[turn_key]
        for csv_col, out_key in agent_cols:
            if out_key not in existing:
                val = _parse_nonzero(row.get(csv_col, ""))
                if val is not None:
                    existing[out_key] = val


def _extract_fields(
    row: dict[str, str], cols: list[tuple[str, str]]
) -> dict[str, float]:
    """Extract nonzero numeric values for the given column mappings."""
    result: dict[str, float] = {}
    for csv_col, out_key in cols:
        val = _parse_nonzero(row.get(csv_col, ""))
        if val is not None:
            result[out_key] = val
    return result


def _parse_nonzero(raw: str) -> Optional[float]:
    """Parse a string as a nonzero float, returning None on failure or zero."""
    if not raw:
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return val if val != 0.0 else None
