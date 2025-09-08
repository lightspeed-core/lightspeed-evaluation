"""Data validation of input data before evaluation."""

from typing import List, Optional

import yaml
from pydantic import ValidationError

from ..models import EvaluationData
from .exceptions import DataValidationError
from .loader import CONVERSATION_LEVEL_METRICS, TURN_LEVEL_METRICS

# Metric requirements mapping
METRIC_REQUIREMENTS = {
    "ragas:faithfulness": {
        "required_fields": ["response", "contexts"],
        "description": "requires 'response' and 'contexts' fields",
    },
    "ragas:response_relevancy": {
        "required_fields": ["response"],
        "description": "requires 'response' field",
    },
    "ragas:context_recall": {
        "required_fields": ["response", "contexts", "expected_response"],
        "description": "requires 'response', 'contexts', and 'expected_response' fields",
    },
    "ragas:context_relevance": {
        "required_fields": ["response", "contexts"],
        "description": "requires 'response' and 'contexts' fields",
    },
    "ragas:context_precision_with_reference": {
        "required_fields": ["response", "contexts", "expected_response"],
        "description": "requires 'response', 'contexts', and 'expected_response' fields",
    },
    "ragas:context_precision_without_reference": {
        "required_fields": ["response", "contexts"],
        "description": "requires 'response' and 'contexts' fields",
    },
    "custom:answer_correctness": {
        "required_fields": ["response", "expected_response"],
        "description": "requires 'response' and 'expected_response' fields",
    },
    "custom:tool_eval": {
        "required_fields": ["tool_calls", "expected_tool_calls"],
        "description": (
            "requires 'tool_calls' and 'expected_tool_calls' fields "
            "with 'tool_name' and 'arguments'"
        ),
    },
}


def format_pydantic_error(error: ValidationError) -> str:
    """Format Pydantic validation error for better readability."""
    errors = []
    for err in error.errors():
        field = " -> ".join(str(loc) for loc in err["loc"])
        message = err["msg"]
        errors.append(f"{field}: {message}")
    return "; ".join(errors)


class DataValidator:
    """Data validator for evaluation data."""

    def __init__(self, api_enabled: bool = False) -> None:
        """Initialize validator."""
        self.validation_errors: List[str] = []
        self.evaluation_data: Optional[List[EvaluationData]] = None
        self.api_enabled = api_enabled
        self.original_data_path: Optional[str] = None

    def load_evaluation_data(self, data_path: str) -> List[EvaluationData]:
        """Load and validate evaluation data from YAML file."""
        self.original_data_path = data_path

        try:
            with open(data_path, "r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except FileNotFoundError as exc:
            raise DataValidationError(
                f"Evaluation data file not found: {data_path}"
            ) from exc
        except yaml.YAMLError as e:
            raise DataValidationError(f"Invalid YAML syntax in {data_path}: {e}") from e

        # Validate YAML root structure
        if raw_data is None:
            raise DataValidationError("Empty or invalid YAML file")
        if not isinstance(raw_data, list):
            raise DataValidationError(
                f"YAML root must be a list, got {type(raw_data).__name__}"
            )

        # Convert raw data to Pydantic models
        evaluation_data = []
        for i, data_dict in enumerate(raw_data):
            try:
                eval_data = EvaluationData(**data_dict)
                evaluation_data.append(eval_data)
            except ValidationError as e:
                conversation_id = data_dict.get(
                    "conversation_group_id", f"item_{i + 1}"
                )
                error_details = format_pydantic_error(e)
                raise DataValidationError(
                    f"Validation error in conversation '{conversation_id}': {error_details}"
                ) from e
            except Exception as e:
                raise DataValidationError(
                    f"Failed to parse evaluation data item {i + 1}: {e}"
                ) from e

        # Additional validation
        if not self.validate_evaluation_data(evaluation_data):
            raise DataValidationError("Evaluation data validation failed")

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

        validation_msg = "✅ All data validation passed"
        if self.api_enabled:
            validation_msg += " (API mode - data will be enhanced via API)"
        print(validation_msg)
        return True

    def _validate_metrics_availability(self, data: EvaluationData) -> None:
        """Validate that specified metrics are available/supported."""
        conversation_id = data.conversation_group_id

        # Validate turn metrics
        if data.turn_metrics:
            for metric in data.turn_metrics:
                if metric not in TURN_LEVEL_METRICS:
                    self.validation_errors.append(
                        f"Conversation {conversation_id}: Unknown turn metric '{metric}'"
                    )

        # Validate conversation metrics
        if data.conversation_metrics:
            for metric in data.conversation_metrics:
                if metric not in CONVERSATION_LEVEL_METRICS:
                    self.validation_errors.append(
                        f"Conversation {conversation_id}: Unknown conversation metric '{metric}'"
                    )

    def _validate_metric_requirements(self, data: EvaluationData) -> None:
        """Validate that required fields exist for specified metrics."""
        conversation_group_id = data.conversation_group_id

        field_errors = self._check_metric_requirements(data, self.api_enabled)

        # Add conversation group ID prefix to errors
        for error in field_errors:
            self.validation_errors.append(
                f"Conversation {conversation_group_id}: {error}"
            )

    def _check_metric_requirements(
        self, data: EvaluationData, api_enabled: bool = True
    ) -> List[str]:
        """Check that required fields exist for specified metrics and API configuration."""
        errors = []

        # Check each turn against metric requirements
        for turn_data in data.turns:
            # Skip validation if no turn metrics specified
            if not data.turn_metrics:
                continue

            for metric in data.turn_metrics:
                if metric not in METRIC_REQUIREMENTS:
                    continue  # Unknown metrics are handled separately

                requirements = METRIC_REQUIREMENTS[metric]
                required_fields = requirements["required_fields"]
                description = requirements["description"]

                # Check each required field
                for field_name in required_fields:
                    field_value = getattr(turn_data, field_name, None)

                    # For API-populated fields, allow None if API is enabled
                    api_populated_fields = ["response", "contexts", "tool_calls"]
                    if (
                        field_name in api_populated_fields
                        and api_enabled
                        and field_value is None
                    ):
                        continue  # will be populated by API

                    # Check if field is missing or empty
                    if (
                        field_value is None
                        or (isinstance(field_value, str) and not field_value.strip())
                        or (isinstance(field_value, list) and not field_value)
                    ):
                        api_context = (
                            " when API is disabled"
                            if field_name in api_populated_fields and not api_enabled
                            else ""
                        )
                        errors.append(
                            f"TurnData {turn_data.turn_id}: Metric '{metric}' "
                            f"{description}{api_context}"
                        )
                        break  # Only report once per metric per turn

        return errors
