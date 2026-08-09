"""Save eval_report.json."""

import json
import os

from lightspeed_evaluation.pipeline.behavioral.models import EvalReport


def save_report(report: EvalReport, output_dir: str) -> str:
    """Serialize and save eval_report.json.

    Args:
        report: The EvalReport to save.
        output_dir: Directory to write the file (typically eval_<ts>/).

    Returns:
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "eval_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(exclude_none=True), f, indent=2)
    return path
