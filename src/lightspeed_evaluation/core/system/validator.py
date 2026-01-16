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
    "custom:keywords_eval": {
        "required_fields": ["response", "expected_keywords"],
        "description": "requires 'response' and 'expected_keywords' fields",
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

# NLP metrics share identical requirements - add them programmatically
_NLP_METRIC_REQUIREMENTS = {
    "required_fields": ["response", "expected_response"],
    "description": "requires 'response' and 'expected_response' fields",
}
for _nlp_metric in ["nlp:bleu", "nlp:rouge", "nlp:semantic_similarity_distance"]:
    METRIC_REQUIREMENTS[_nlp_metric] = _NLP_METRIC_REQUIREMENTS


def format_pydantic_error(error: ValidationError) -> str:
    """Format Pydantic validation error for better readability."""
    errors = []
    for err in error.errors():
        field = " -> ".join(str(loc) for loc in err["loc"])
        message = err["msg"]
        errors.append(f"{field}: {message}")
    return "; ".join(errors)


class DataValidator:  # pylint: disable=too-few-public-methods
    """Data validator for evaluation data.

    Single entry point: load_evaluation_data() which handles loading,
    validation, and optional script validation.
    """

    def __init__(
        self, api_enabled: bool = False, fail_on_invalid_data: bool = True
    ) -> None:
        """Initialize validator."""
        self.validation_errors: list[str] = []
        self.evaluation_data: Optional[list[EvaluationData]] = None
        self.api_enabled = api_enabled
        self.original_data_path: Optional[str] = None
        self.fail_on_invalid_data = fail_on_invalid_data

    def load_evaluation_data(
        self,
        data_path: str,
        tags: Optional[list[str]] = None,
        conv_ids: Optional[list[str]] = None,
    ) -> list[EvaluationData]:
        """Load, filter, and validate evaluation data from YAML file.

        Filtering logic:
        - no tags, no conv_ids -> return all conversations
        - tags set, no conv_ids -> return conversations with matching tags
        - no tags, conv_ids set -> return conversations with matching IDs
        - both set -> return conversations matching either tag OR conv_id

        Args:
            data_path: Path to the evaluation data YAML file
            tags: Optional list of tags to filter by
            conv_ids: Optional list of conversation group IDs to filter by

        Returns:
            Filtered and validated list of Evaluation Data
        """
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

        # Filter by scope before validation
        evaluation_data = self._filter_by_scope(evaluation_data, tags, conv_ids)

        # Semantic validation (metrics availability and requirements)
        if not self._validate_evaluation_data(evaluation_data):
            raise DataValidationError("Evaluation data validation failed")

        # Validate scripts only if API is enabled
        if self.api_enabled:
            self._validate_scripts(evaluation_data)

        self.evaluation_data = evaluation_data
        return evaluation_data

    def _filter_by_scope(
        self,
        evaluation_data: list[EvaluationData],
        tags: Optional[list[str]] = None,
        conv_ids: Optional[list[str]] = None,
    ) -> list[EvaluationData]:
        """Filter evaluation data based on tags and/or conversation group IDs.

        Args:
            evaluation_data: List of conversation group data to filter
            tags: Optional list of tags to filter by
            conv_ids: Optional list of conversation group IDs to filter by

        Returns:
            Filtered list of Evaluation Data matching the criteria
        """
        total_count = len(evaluation_data)

        if not tags and not conv_ids:
            print(f"📋 Evaluation data loaded: {total_count} conversations")
            return evaluation_data

        tag_set = set(tags) if tags else set()
        conv_id_set = set(conv_ids) if conv_ids else set()

        filtered = [
            conv_data
            for conv_data in evaluation_data
            if conv_data.tag in tag_set
            or conv_data.conversation_group_id in conv_id_set
        ]

        print(
            f"📋 Evaluation data: {len(filtered)} of {total_count} "
            "conversations (filtered)"
        )
        return filtered

    def _validate_evaluation_data(self, evaluation_data: list[EvaluationData]) -> bool:
        """Validate metrics availability and requirements for all evaluation data."""
        self.validation_errors = []

        for data in evaluation_data:
            self._validate_metrics_availability(data)
            self._validate_metric_requirements(data)

        if self.validation_errors:
            print("❌ Validation Errors:")
            for error in self.validation_errors:
                print(f"  • {error}")

            if self.fail_on_invalid_data:
                return False

            print("❌ Validation Errors!, ignoring as instructed")
            return True

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
                        turn_data.add_invalid_metric(metric)
                        self.validation_errors.append(
                            f"Conversation {conversation_id}, Turn {turn_data.turn_id}: "
                            f"Unknown turn metric '{metric}'"
                        )

        # Validate conversation metrics
        if data.conversation_metrics:
            for metric in data.conversation_metrics:
                if metric not in CONVERSATION_LEVEL_METRICS:
                    data.add_invalid_metric(metric)
                    self.validation_errors.append(
                        f"Conversation {conversation_id}: Unknown conversation metric '{metric}'"
                    )

    def _validate_metric_requirements(self, data: EvaluationData) -> None:
        """Validate that required fields exist for specified metrics."""
        conversation_group_id = data.conversation_group_id

        field_errors = self._check_metric_requirements(data, self.api_enabled)

        # No errors
        if not field_errors:
            return

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
                        turn_data.add_invalid_metric(metric)

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
