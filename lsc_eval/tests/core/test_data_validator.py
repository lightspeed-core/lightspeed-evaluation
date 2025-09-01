"""Test cases for DataValidator class based on system.yaml configuration."""

import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch, MagicMock

import pytest
import yaml

from lsc_eval.core.data_validator import DataValidator
from lsc_eval.core.models import EvaluationData, TurnData
from lsc_eval.core.config_loader import populate_metric_mappings


class TestDataValidator:
    """Test DataValidator class functionality."""

    def test_data_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()

        assert validator.validation_errors == []
        assert validator.evaluation_data is None

    def test_load_evaluation_data_success(
        self, 
        evaluation_data_file: Path, 
        sample_system_config: Dict[str, Any]
    ):
        """Test successful evaluation data loading."""
        # Setup metric mappings first
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        validator = DataValidator()
        
        with patch('builtins.print'):  # Suppress print statements
            evaluation_data = validator.load_evaluation_data(str(evaluation_data_file))

        assert len(evaluation_data) == 2
        assert all(isinstance(data, EvaluationData) for data in evaluation_data)
        assert validator.evaluation_data == evaluation_data

        # Check first conversation group
        assert evaluation_data[0].conversation_group_id == "conv_group_1"
        assert len(evaluation_data[0].turns) == 1
        assert evaluation_data[0].turns[0].turn_id == 1

        # Check second conversation group
        assert evaluation_data[1].conversation_group_id == "conv_group_2"
        assert len(evaluation_data[1].turns) == 2

    def test_load_evaluation_data_file_not_found(self):
        """Test evaluation data loading with non-existent file."""
        validator = DataValidator()

        with pytest.raises(FileNotFoundError):
            validator.load_evaluation_data("/nonexistent/file.yaml")

    def test_load_evaluation_data_invalid_yaml(self, temp_dir: Path):
        """Test evaluation data loading with invalid YAML."""
        invalid_file = temp_dir / "invalid.yaml"
        invalid_file.write_text("invalid: yaml: content: [")

        validator = DataValidator()

        with pytest.raises(yaml.YAMLError):
            validator.load_evaluation_data(str(invalid_file))

    def test_load_evaluation_data_invalid_structure(self, temp_dir: Path):
        """Test evaluation data loading with invalid data structure."""
        invalid_data = [
            {
                "conversation_group_id": "",  # Invalid: empty ID
                "turns": []  # Invalid: empty turns
            }
        ]

        invalid_file = temp_dir / "invalid_structure.yaml"
        with open(invalid_file, "w") as f:
            yaml.dump(invalid_data, f)

        validator = DataValidator()

        with pytest.raises(ValueError, match="Data validation failed"):
            validator.load_evaluation_data(str(invalid_file))

    def test_load_evaluation_data_validation_failure(
        self, 
        temp_dir: Path,
        sample_system_config: Dict[str, Any]
    ):
        """Test evaluation data loading with validation failures."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        # Create data with validation issues
        invalid_data = [
            {
                "conversation_group_id": "conv_1",
                "turn_metrics": ["invalid:metric"],  # Unknown metric
                "turns": [
                    {
                        "turn_id": 1,
                        "query": "Test query",
                        "response": "Test response"
                    }
                ]
            }
        ]

        invalid_file = temp_dir / "validation_failure.yaml"
        with open(invalid_file, "w") as f:
            yaml.dump(invalid_data, f)

        validator = DataValidator()

        with pytest.raises(ValueError, match="Evaluation data validation failed"):
            validator.load_evaluation_data(str(invalid_file))

    def test_validate_evaluation_data_success(self, sample_system_config: Dict[str, Any]):
        """Test successful evaluation data validation."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        # Create valid evaluation data
        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response",
            contexts=[{"content": "Test context"}],
            expected_response="Expected response"
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_1",
            turn_metrics=["ragas:faithfulness", "ragas:response_relevancy"],
            conversation_metrics=["deepeval:conversation_completeness"],
            turns=[turn_data]
        )

        validator = DataValidator()
        
        with patch('builtins.print'):  # Suppress print statements
            result = validator.validate_evaluation_data([eval_data])

        assert result is True
        assert len(validator.validation_errors) == 0

    def test_validate_evaluation_data_unknown_metrics(self, sample_system_config: Dict[str, Any]):
        """Test validation with unknown metrics."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_1",
            turn_metrics=["unknown:turn_metric", "ragas:response_relevancy"],  # Use metric that doesn't require contexts
            conversation_metrics=["unknown:conversation_metric"],
            turns=[turn_data]
        )

        validator = DataValidator()
        
        with patch('builtins.print'):  # Suppress print statements
            result = validator.validate_evaluation_data([eval_data])

        assert result is False
        assert len(validator.validation_errors) == 2
        assert "Unknown turn metric 'unknown:turn_metric'" in validator.validation_errors[0]
        assert "Unknown conversation metric 'unknown:conversation_metric'" in validator.validation_errors[1]

    def test_validate_evaluation_data_metric_requirements(self, sample_system_config: Dict[str, Any]):
        """Test validation of metric requirements."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        # Turn without contexts but requiring contexts
        turn_without_contexts = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
            # No contexts
        )

        eval_data = EvaluationData(
            conversation_group_id="conv_1",
            turn_metrics=["ragas:faithfulness"],  # Requires contexts
            turns=[turn_without_contexts]
        )

        validator = DataValidator()
        
        with patch('builtins.print'):  # Suppress print statements
            result = validator.validate_evaluation_data([eval_data])

        assert result is False
        assert len(validator.validation_errors) > 0
        assert "Metric 'ragas:faithfulness' requires contexts" in validator.validation_errors[0]

    def test_validate_evaluation_data_multiple_conversations(self, sample_system_config: Dict[str, Any]):
        """Test validation with multiple conversation groups."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        # Valid conversation
        valid_turn = TurnData(
            turn_id=1,
            query="Valid query",
            response="Valid response",
            contexts=[{"content": "Valid context"}]
        )

        valid_eval = EvaluationData(
            conversation_group_id="valid_conv",
            turn_metrics=["ragas:faithfulness"],
            turns=[valid_turn]
        )

        # Invalid conversation
        invalid_turn = TurnData(
            turn_id=1,
            query="Invalid query",
            response="Invalid response"
            # Missing contexts
        )

        invalid_eval = EvaluationData(
            conversation_group_id="invalid_conv",
            turn_metrics=["ragas:faithfulness"],  # Requires contexts
            turns=[invalid_turn]
        )

        validator = DataValidator()
        
        with patch('builtins.print'):  # Suppress print statements
            result = validator.validate_evaluation_data([valid_eval, invalid_eval])

        assert result is False
        assert len(validator.validation_errors) > 0
        assert "invalid_conv" in validator.validation_errors[0]

    def test_validate_metrics_availability_turn_metrics(self, sample_system_config: Dict[str, Any]):
        """Test _validate_metrics_availability for turn metrics."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=["ragas:faithfulness", "unknown:metric"],
            turns=[turn_data]
        )

        validator = DataValidator()
        validator._validate_metrics_availability(eval_data)

        assert len(validator.validation_errors) == 1
        assert "Unknown turn metric 'unknown:metric'" in validator.validation_errors[0]
        assert "test_conv" in validator.validation_errors[0]

    def test_validate_metrics_availability_conversation_metrics(self, sample_system_config: Dict[str, Any]):
        """Test _validate_metrics_availability for conversation metrics."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        turn_data = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            conversation_metrics=["deepeval:conversation_completeness", "unknown:conv_metric"],
            turns=[turn_data]
        )

        validator = DataValidator()
        validator._validate_metrics_availability(eval_data)

        assert len(validator.validation_errors) == 1
        assert "Unknown conversation metric 'unknown:conv_metric'" in validator.validation_errors[0]
        assert "test_conv" in validator.validation_errors[0]

    def test_validate_metric_requirements_contexts_required(self):
        """Test _validate_metric_requirements for context-required metrics."""
        turn_without_contexts = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response"
        )

        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=[
                "ragas:faithfulness",
                "ragas:context_recall",
                "ragas:context_relevance"
            ],
            turns=[turn_without_contexts]
        )

        validator = DataValidator()
        validator._validate_metric_requirements(eval_data)

        # Should have errors for all context-required metrics
        context_required_errors = [
            error for error in validator.validation_errors 
            if "requires contexts" in error
        ]
        assert len(context_required_errors) == 3

    def test_validate_metric_requirements_expected_response_required(self):
        """Test _validate_metric_requirements for expected_response-required metrics."""
        turn_without_expected = TurnData(
            turn_id=1,
            query="Test query",
            response="Test response",
            contexts=[{"content": "Context"}]
        )

        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=[
                "ragas:context_recall",
                "custom:answer_correctness"
            ],
            turns=[turn_without_expected]
        )

        validator = DataValidator()
        validator._validate_metric_requirements(eval_data)

        # Should have errors for expected_response-required metrics
        expected_required_errors = [
            error for error in validator.validation_errors 
            if "requires expected_response" in error
        ]
        assert len(expected_required_errors) == 2

    def test_validate_metric_requirements_multiple_turns(self):
        """Test _validate_metric_requirements with multiple turns."""
        # First turn missing contexts
        turn1 = TurnData(
            turn_id=1,
            query="Query 1",
            response="Response 1"
        )

        # Second turn missing expected_response
        turn2 = TurnData(
            turn_id=2,
            query="Query 2",
            response="Response 2",
            contexts=[{"content": "Context"}]
        )

        eval_data = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=["ragas:faithfulness", "custom:answer_correctness"],
            turns=[turn1, turn2]
        )

        validator = DataValidator()
        validator._validate_metric_requirements(eval_data)

        # Should have errors for both turns
        turn1_errors = [error for error in validator.validation_errors if "TurnData 1:" in error]
        turn2_errors = [error for error in validator.validation_errors if "TurnData 2:" in error]
        
        assert len(turn1_errors) == 2  # Missing contexts and expected_response
        assert len(turn2_errors) == 1  # Missing expected_response only

    def test_validation_error_handling(self, sample_system_config: Dict[str, Any]):
        """Test validation error accumulation and clearing."""
        # Setup metric mappings
        populate_metric_mappings(sample_system_config["metrics_metadata"])

        turn_data = TurnData(turn_id=1, query="Test query", response="Test response")
        
        # Test error accumulation
        invalid_eval = EvaluationData(
            conversation_group_id="test_conv",
            turn_metrics=["unknown:metric", "ragas:faithfulness"],  # Unknown + requires contexts
            conversation_metrics=["unknown:conv_metric"],
            turns=[turn_data]
        )

        validator = DataValidator()
        
        with patch('builtins.print'):
            result1 = validator.validate_evaluation_data([invalid_eval])

        assert result1 is False
        assert len(validator.validation_errors) >= 3

        # Test error clearing between runs
        valid_eval = EvaluationData(
            conversation_group_id="valid_conv",
            turn_metrics=["ragas:response_relevancy"],
            turns=[turn_data]
        )

        with patch('builtins.print'):
            result2 = validator.validate_evaluation_data([valid_eval])

        assert result2 is True
        assert len(validator.validation_errors) == 0


