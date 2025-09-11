"""Simple data persistence utilities for evaluation framework."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from ..constants import DEFAULT_OUTPUT_DIR
from ..models import EvaluationData


# Use caching
def save_evaluation_data(
    evaluation_data: list[EvaluationData],
    original_data_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Optional[str]:
    """Save updated evaluation data, backing up original with timestamp."""
    original_path = Path(original_data_path)
    backup_path = None

    try:
        output_path = Path(output_dir)

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Create backup with timestamp in output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = (
            output_path / f"{original_path.stem}_bkp_{timestamp}{original_path.suffix}"
        )

        # Copy original to backup
        shutil.copy2(str(original_path), str(backup_path))
        print(f"📋 Original file backed up: {backup_path}")

        # Save updated data to original filename
        with open(original_path, "w", encoding="utf-8") as f:
            yaml.dump(
                [conv_data.model_dump() for conv_data in evaluation_data],
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=2,
            )

        print(f"💾 Updated evaluation data saved: {original_path}")
        print(f"📁 Backup created: {backup_path}")
        return str(original_path)

    except (OSError, yaml.YAMLError) as e:
        print(f"❌ Failed to save updated evaluation data: {e}")
        print(f"💾 Original file remains intact, backup available at: {backup_path}")
        return None
