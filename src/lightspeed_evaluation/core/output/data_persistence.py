"""Simple data persistence utilities for evaluation framework."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from lightspeed_evaluation.core.constants import DEFAULT_OUTPUT_DIR
from lightspeed_evaluation.core.models import EvaluationData
from lightspeed_evaluation.core.models.data import DatasetMetadata


def save_evaluation_data(
    evaluation_data: list[EvaluationData],
    original_data_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    dataset_metadata: Optional[DatasetMetadata] = None,
) -> Optional[str]:
    """Save amended evaluation data to output directory with timestamp.

    When *dataset_metadata* is provided the file is written in the dict
    format (``metadata`` + ``conversations`` keys) so that dataset-level
    metadata is preserved across amend cycles.  Without metadata the
    original list format is used for backward compatibility.
    """
    original_path = Path(original_data_path)
    amended_data_path = None

    try:
        output_path = Path(output_dir)

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        # Create amended data file with timestamp in output directory
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        amended_data_path = (
            output_path
            / f"{original_path.stem}_amended_{timestamp}{original_path.suffix}"
        )

        conversations = [
            conv_data.model_dump(mode="json") for conv_data in evaluation_data
        ]

        output_data: Any = conversations
        if dataset_metadata is not None:
            output_data = {
                "metadata": dataset_metadata.model_dump(mode="json", exclude_none=True),
                "conversations": conversations,
            }

        # Save amended data to output directory
        with open(amended_data_path, "w", encoding="utf-8") as f:
            yaml.dump(
                output_data,
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
