"""Data validation of input data before evaluation."""

import os
from pathlib import Path
from typing import Optional, Union

import yaml
from pydantic import ValidationError

from lightspeed_evaluation.core.models import EvaluationData
from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.core.system.loader import (
    CONVERSATION_LEVEL_METRICS,
    TURN_LEVEL_METRICS,
)
from lightspeed_evaluation.core.utils import sanitize_run_name

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
    "custom:intent_eval": {
        "required_fields": ["response", "expected_intent"],
        "description": "requires 'response' and 'expected_intent' fields",
    },
    "custom:tool_eval": {
        "required_fields": ["tool_calls", "expected_tool_calls"],
        "description": (
            "requires 'tool_calls' and 'expected_tool_calls' fields "
            "with 'tool_name' and 'arguments'"
        ),
    },
    "script:action_eval": {
        "required_fields": ["verify_script"],
        "description": "requires 'verify_script' field",
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
        self.validation_errors: list[str] = []
        self.evaluation_data: Optional[list[EvaluationData]] = None
        self.api_enabled = api_enabled
        self.original_data_path: Optional[str] = None

    def load_evaluation_data(self, data_path: str) -> list[EvaluationData]:
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
                # Set default run_name from YAML filename if not provided
                if "run_name" not in data_dict or data_dict["run_name"] is None:
                    yaml_filename = Path(data_path).stem  # Get filename without extension
                    data_dict["run_name"] = sanitize_run_name(yaml_filename)

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

        # Validate scripts only if API is enabled
        if self.api_enabled:
            self._validate_scripts(evaluation_data)

        self.evaluation_data = evaluation_data
        print(f"📋 Evaluation data loaded: {len(evaluation_data)} conversations")

        return evaluation_data

    def validate_evaluation_data(self, evaluation_data: list[EvaluationData]) -> bool:
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

        # Validate per-turn metrics
        for turn_data in data.turns:
            if turn_data.turn_metrics:
                for metric in turn_data.turn_metrics:
                    if metric not in TURN_LEVEL_METRICS:
                        self.validation_errors.append(
                            f"Conversation {conversation_id}, Turn {turn_data.turn_id}: "
                            f"Unknown turn metric '{metric}'"
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
    ) -> list[str]:
        """Check that required fields exist for specified metrics and API configuration."""
        errors = []

        # Check each turn against metric requirements
        for turn_data in data.turns:
            # Skip validation if no turn metrics specified
            if not turn_data.turn_metrics:
                continue

            for metric in turn_data.turn_metrics:
                if metric not in METRIC_REQUIREMENTS:
                    continue  # Unknown metrics are handled separately

                # Skip script metric validation if API is disabled
                if metric.startswith("script:") and not self.api_enabled:
                    continue

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

    def _validate_scripts(self, evaluation_data: list[EvaluationData]) -> None:
        """Validate all script paths when API is enabled."""
        for data in evaluation_data:
            # Validate conversation-level scripts
            data.setup_script = self._validate_single_script(
                data.setup_script, "Setup", data.conversation_group_id
            )
            data.cleanup_script = self._validate_single_script(
                data.cleanup_script, "Cleanup", data.conversation_group_id
            )

            # Validate turn-level scripts
            for turn in data.turns:
                turn.verify_script = self._validate_single_script(
                    turn.verify_script,
                    "Verify",
                    f"{data.conversation_group_id}, Turn {turn.turn_id}",
                )

    def _validate_single_script(
        self,
        script_file: Optional[Union[str, Path]],
        script_type: str,
        context: str,
    ) -> Optional[Path]:
        """Validate a single script file and return the validated Path object."""
        if script_file is None:
            return None

        if isinstance(script_file, str):
            script_file = Path(script_file)

        # Expand user home directory shortcuts
        script_file = script_file.expanduser()

        # Resolve relative paths against the YAML file directory, not CWD
        if not script_file.is_absolute() and self.original_data_path:
            yaml_dir = Path(self.original_data_path).parent
            script_file = (yaml_dir / script_file).resolve()
        else:
            script_file = script_file.resolve()

        # Validate existence and file type
        if not script_file.exists():
            raise DataValidationError(
                f"Conversation {context}: {script_type} script not found: {script_file}"
            )

        if not script_file.is_file():
            raise DataValidationError(
                f"Conversation {context}: {script_type} script path is not a file: {script_file}"
            )

        # Check if script is executable or can be made executable
        if not os.access(script_file, os.X_OK):
            try:
                script_file.chmod(0o755)
            except (OSError, PermissionError) as exc:
                raise DataValidationError(
                    f"Conversation {context}: {script_type} script is not executable: {script_file}"
                ) from exc

        return script_file
