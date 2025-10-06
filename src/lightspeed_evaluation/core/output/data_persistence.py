"""Simple data persistence utilities for evaluation framework."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from lightspeed_evaluation.core.constants import DEFAULT_OUTPUT_DIR
from lightspeed_evaluation.core.models import EvaluationData


# Use caching
def save_evaluation_data(
    evaluation_data: list[EvaluationData],
    original_data_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Optional[str]:
    """Save amended evaluation data to output directory with timestamp."""
    original_path = Path(original_data_path)
    amended_data_path = None

    try:
        output_path = Path(output_dir)

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Create amended data file with timestamp in output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        amended_data_path = (
            output_path
            / f"{original_path.stem}_amended_{timestamp}{original_path.suffix}"
        )

        # Save amended data to output directory
        with open(amended_data_path, "w", encoding="utf-8") as f:
            yaml.dump(
                [conv_data.model_dump() for conv_data in evaluation_data],
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
                indent=2,
            )

        print(f"💾 Amended evaluation data saved to: {amended_data_path}")
        return str(amended_data_path)

    except (OSError, yaml.YAMLError) as e:
        print(f"❌ Failed to save amended evaluation data: {e}")
        return None
