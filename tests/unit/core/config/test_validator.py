"""Unit tests for core.config.validator module."""

import tempfile
from unittest.mock import patch

import pytest
import yaml

from lightspeed_evaluation.core.config.validator import DataValidator
from lightspeed_evaluation.core.config.models import EvaluationData, TurnData


class TestDataValidator:
    """Unit tests for DataValidator class."""

    def test_data_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()

        assert validator.validation_errors == []
        assert validator.evaluation_data is None

    def test_load_evaluation_data_from_yaml_success(self):
        """Test successful loading of evaluation data from YAML."""
        eval_data = [
            {
                "conversation_group_id": "test_conv",
                "turn_metrics": ["ragas:faithfulness"],
                "conversation_metrics": [],
                "turns": [
                    {
                        "turn_id": 1,
                        "query": "Test query",
                        "response": "Test response",
                        "contexts": [
                            {"content": "Test context for faithfulness metric"}
                        ],
                    }
                ],
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(eval_data, f)
            data_path = f.name

        try:
            # Mock the metric validation to avoid validation errors
            with patch(
                "lightspeed_evaluation.core.config.validator.TURN_LEVEL_METRICS",
                {"ragas:faithfulness"},
            ):
                with patch(
                    "lightspeed_evaluation.core.config.validator.CONVERSATION_LEVEL_METRICS",
                    set(),
                ):
                    validator = DataValidator()
                    loaded_data = validator.load_evaluation_data(data_path)

                    assert len(loaded_data) == 1
                    assert loaded_data[0].conversation_group_id == "test_conv"
                    assert len(loaded_data[0].turns) == 1
                    assert loaded_data[0].turns[0].turn_id == 1

        finally:
            import os

            os.unlink(data_path)

    def test_load_evaluation_data_file_not_found(self):
        """Test loading evaluation data from non-existent file."""
        validator = DataValidator()

        with pytest.raises(FileNotFoundError):
            validator.load_evaluation_data("nonexistent_file.yaml")

    def test_load_evaluation_data_invalid_yaml(self):
        """Test loading evaluation data from invalid YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            data_path = f.name

        try:
            validator = DataValidator()
            with pytest.raises(yaml.YAMLError):
                validator.load_evaluation_data(data_path)

        finally:
            import os

            os.unlink(data_path)

    def test_load_evaluation_data_invalid_structure(self):
        """Test loading evaluation data with invalid structure."""
        # Missing required fields
        eval_data = [{"invalid_field": "value"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(eval_data, f)
            data_path = f.name

        try:
            validator = DataValidator()
            with pytest.raises((ValueError, KeyError)):
                validator.load_evaluation_data(data_path)

        finally:
            import os

            os.unlink(data_path)

    def test_validate_evaluation_data_success(self):
        """Test successful validation of evaluation data."""
        turn = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response",
            contexts=[{"content": "Test context"}],
        )

        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=["ragas:faithfulness"],
            conversation_metrics=[],
            turns=[turn],
        )

        validator = DataValidator()

        # Mock the metric sets to include our test metric
        with patch(
            "lightspeed_evaluation.core.config.validator.TURN_LEVEL_METRICS",
            {"ragas:faithfulness"},
        ):
            with patch(
                "lightspeed_evaluation.core.config.validator.CONVERSATION_LEVEL_METRICS",
                set(),
            ):
                errors = validator.validate_evaluation_data([eval_data])

                assert errors is True

    def test_validate_evaluation_data_unknown_metric(self):
        """Test validation with unknown metric."""
        turn = TurnData(turn_id=1, query="Test query", response="Test response")

        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=["unknown:metric"],
            conversation_metrics=[],
            turns=[turn],
        )

        validator = DataValidator()

        # Mock empty metric sets
        with patch(
            "lightspeed_evaluation.core.config.validator.TURN_LEVEL_METRICS", set()
        ):
            with patch(
                "lightspeed_evaluation.core.config.validator.CONVERSATION_LEVEL_METRICS",
                set(),
            ):
                errors = validator.validate_evaluation_data([eval_data])

                assert errors is False

    def test_validate_evaluation_data_empty_list(self):
        """Test validation of empty evaluation data list."""
        validator = DataValidator()
        errors = validator.validate_evaluation_data([])

        assert errors is True  # Empty list should not cause errors

    def test_validation_errors_accumulation(self):
        """Test that validation errors are properly accumulated."""
        validator = DataValidator()

        # Initially empty
        assert len(validator.validation_errors) == 0

        # Add some mock errors
        validator.validation_errors.append("Error 1")
        validator.validation_errors.append("Error 2")

        assert len(validator.validation_errors) == 2
        assert "Error 1" in validator.validation_errors
        assert "Error 2" in validator.validation_errors

    def test_clear_validation_errors(self):
        """Test clearing validation errors."""
        validator = DataValidator()
        validator.validation_errors = ["Error 1", "Error 2"]

        # Clear errors
        validator.validation_errors.clear()

        assert len(validator.validation_errors) == 0

    def test_load_evaluation_data_multiple_conversations(self):
        """Test loading multiple conversation groups."""
        eval_data = [
            {
                "conversation_group_id": "conv1",
                "turn_metrics": [],
                "conversation_metrics": [],
                "turns": [{"turn_id": 1, "query": "Query 1", "response": "Response 1"}],
            },
            {
                "conversation_group_id": "conv2",
                "turn_metrics": [],
                "conversation_metrics": [],
                "turns": [{"turn_id": 1, "query": "Query 2", "response": "Response 2"}],
            },
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(eval_data, f)
            data_path = f.name

        try:
            validator = DataValidator()
            loaded_data = validator.load_evaluation_data(data_path)

            assert len(loaded_data) == 2
            assert loaded_data[0].conversation_group_id == "conv1"
            assert loaded_data[1].conversation_group_id == "conv2"

        finally:
            import os

            os.unlink(data_path)
