"""Agent Goal Eval data management."""

from pathlib import Path
from typing import Any

import yaml

from ..utils.exceptions import ConfigurationError
from .models import EvaluationDataConfig


class AgentGoalEvalDataManager:
    """Processes agent eval data and validation."""

    def __init__(self, eval_data_file: str):
        """Initialize configuration manager."""
        self.eval_data_file = Path(eval_data_file)
        self.eval_data: list[EvaluationDataConfig] = []
        self._validate_eval_data_file()
        self._load_eval_data()

    def _validate_eval_data_file(self) -> None:
        """Validate eval data file exists and is readable."""
        if not self.eval_data_file.exists():
            raise ConfigurationError(f"Eval data file not found: {self.eval_data_file}")

        if not self.eval_data_file.is_file():
            raise ConfigurationError(
                f"Eval data file path is not a file: {self.eval_data_file}"
            )

    def _load_eval_data(self) -> None:
        """Load evaluation data from YAML file."""
        try:
            with open(self.eval_data_file, "r", encoding="utf-8") as file:
                eval_data = yaml.safe_load(file)

            if not isinstance(eval_data, list):
                raise ConfigurationError(
                    "Eval data file must contain a list of evaluations"
                )

            self.eval_data = []
            for data in eval_data:
                self._validate_eval_data(data)
                self.eval_data.append(EvaluationDataConfig(**data))

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in eval data file: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading eval data file: {e}") from e

    def _validate_eval_data(self, eval_data: dict[str, Any]) -> None:
        """Validate a single evaluation data point."""
        required_fields = ["eval_id", "eval_query"]
        for field in required_fields:
            if field not in eval_data:
                raise ConfigurationError(
                    f"Missing required field '{field}' in evaluation data"
                )

        eval_type = eval_data.get("eval_type", "judge-llm")
        if eval_type not in ["judge-llm", "script", "sub-string"]:
            raise ConfigurationError(f"Invalid eval_type: {eval_type}")

        # Validate type-specific requirements
        if eval_type == "judge-llm" and "expected_response" not in eval_data:
            raise ConfigurationError(
                "eval_type 'judge-llm' requires 'expected_response' field"
            )

        if eval_type == "sub-string" and "expected_key_words" not in eval_data:
            raise ConfigurationError(
                "eval_type 'sub-string' requires 'expected_key_words' field"
            )

        if eval_type == "script" and "eval_verify_script" not in eval_data:
            raise ConfigurationError(
                "eval_type 'script' requires 'eval_verify_script' field"
            )

    def get_eval_data(self) -> list[EvaluationDataConfig]:
        """Get all evaluation configurations."""
        return self.eval_data

    def get_eval_count(self) -> int:
        """Get the number of evaluation configurations."""
        return len(self.eval_data)
