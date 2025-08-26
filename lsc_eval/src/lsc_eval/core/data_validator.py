"""Data validation of input data before evaluation."""

from typing import List

import yaml

from .models import EvaluationData
from .config_loader import CONVERSATION_LEVEL_METRICS, TURN_LEVEL_METRICS


class DataValidator:
    """Data validator for evaluation data."""

    def __init__(self):
        """Initialize validator."""
        self.validation_errors = []
        self.evaluation_data = None

    def load_evaluation_data(self, data_path: str) -> List[EvaluationData]:
        """Load and validate evaluation data from YAML file."""
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        # Convert raw data to Pydantic models
        try:
            evaluation_data = [EvaluationData(**item) for item in raw_data]
        except Exception as e:
            raise ValueError(f"Data validation failed: {str(e)}") from e

        # Additional validation
        if not self.validate_evaluation_data(evaluation_data):
            raise ValueError("Evaluation data validation failed")

        self.evaluation_data = evaluation_data
        print(f"📋 Evaluation data loaded: {len(evaluation_data)} conversations")

        return evaluation_data

    def validate_evaluation_data(self, evaluation_data: List[EvaluationData]) -> bool:
        """Validate all evaluation data."""
        self.validation_errors = []

        for data in evaluation_data:
            self._validate_metrics_availability(data)
            self._validate_metric_requirements(data)

        if self.validation_errors:
            print("❌ Validation Errors:")
            for error in self.validation_errors:
                print(f"  • {error}")
            return False

        print("✅ All data validation passed")
        return True

    def _validate_metrics_availability(self, data: EvaluationData):
        """Validate that specified metrics are available/supported."""
        conversation_id = data.conversation_group_id

        # Validate turn metrics
        for metric in data.turn_metrics:
            if metric not in TURN_LEVEL_METRICS:
                self.validation_errors.append(
                    f"Conversation {conversation_id}: Unknown turn metric '{metric}'"
                )

        # Validate conversation metrics
        for metric in data.conversation_metrics:
            if metric not in CONVERSATION_LEVEL_METRICS:
                self.validation_errors.append(
                    f"Conversation {conversation_id}: Unknown conversation metric '{metric}'"
                )

    def _validate_metric_requirements(self, data: EvaluationData):
        """Validate that required fields exist for specified metrics."""
        conversation_id = data.conversation_group_id

        field_errors = data.validate_metric_requirements()

        # Add conversation ID prefix to errors
        for error in field_errors:
            self.validation_errors.append(f"Conversation {conversation_id}: {error}")
