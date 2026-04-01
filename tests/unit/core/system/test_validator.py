# pylint: disable=protected-access

"""Unit tests for core system validator module."""

import tempfile
from pathlib import Path

import pytest

from pydantic import ValidationError

from lightspeed_evaluation.core.models import EvaluationData, SystemConfig, TurnData
from lightspeed_evaluation.core.system.exceptions import DataValidationError
from lightspeed_evaluation.core.system.validator import (
    DataValidator,
    check_metric_required_data,
    format_pydantic_error,
)


class TestFormatPydanticError:
    """Unit tests for format_pydantic_error helper function."""

    def test_format_single_error(self) -> None:
        """Test formatting a single Pydantic validation error."""
        try:
            TurnData(turn_id="1", query="", response="Valid")
        except ValidationError as e:
            formatted = format_pydantic_error(e)
            assert "query" in formatted
            assert "at least 1 character" in formatted

    def test_format_multiple_errors(self) -> None:
        """Test formatting multiple validation errors."""
        try:
            TurnData(turn_id="", query="", response="")
        except ValidationError as e:
            formatted = format_pydantic_error(e)
            # Should contain errors separated by semicolons
            assert ";" in formatted
            assert "turn_id" in formatted or "query" in formatted


class TestCheckMetricRequiredData:
    """Unit tests for check_metric_required_data (runtime required-data check)."""

    def test_all_required_fields_present_returns_ok(self) -> None:
        """When all required fields are present and non-empty, returns (True, "")."""
        turn = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            contexts=["C"],
        )
        ok, msg = check_metric_required_data(turn, "ragas:faithfulness")
        assert ok is True
        assert msg == ""

    def test_missing_contexts_returns_error(self) -> None:
        """When required field contexts is None, returns (False, message)."""
        turn = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            contexts=None,
        )
        ok, msg = check_metric_required_data(turn, "ragas:faithfulness")
        assert ok is False
        assert msg is not None
        assert "contexts" in msg
        assert "missing or empty" in msg

    def test_empty_list_required_field_returns_error(self) -> None:
        """When required field expected_keywords is [], returns (False, message)."""
        turn = TurnData(
            turn_id="1",
            query="Q",
            response="R",
            expected_keywords=[],  # Empty list for keywords_eval
        )
        ok, msg = check_metric_required_data(turn, "custom:keywords_eval")
        assert ok is False
        assert msg is not None
        assert "expected_keywords" in msg

    def test_missing_response_returns_error(self) -> None:
        """When required field response is None, returns (False, message)."""
        turn = TurnData(
            turn_id="1",
            query="Q",
            response=None,
        )
        ok, msg = check_metric_required_data(turn, "ragas:response_relevancy")
        assert ok is False
        assert msg is not None
        assert "response" in msg

    def test_whitespace_only_response_returns_error(self) -> None:
        """When required field response is whitespace-only, returns (False, message)."""
        turn = TurnData(
            turn_id="1",
            query="Q",
            response="   ",
        )
        ok, msg = check_metric_required_data(turn, "ragas:response_relevancy")
        assert ok is False
        assert msg is not None

    def test_unknown_metric_returns_ok(self) -> None:
        """When metric is not in METRIC_REQUIREMENTS, returns (True, None)."""
        turn = TurnData(turn_id="1", query="Q", response="R")
        ok, msg = check_metric_required_data(turn, "geval:some_criteria")
        assert ok is True
        assert msg == ""


class TestDataValidator:
    """Unit tests for DataValidator."""

    def test_validate_evaluation_data_valid(self) -> None:
        """Test validation passes with valid data."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="What is Python?",
            response="Python is a programming language.",
            contexts=["Context"],
            expected_response="Expected",
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is True
        assert len(validator.validation_errors) == 0

    def test_validate_metrics_availability_unknown_turn_metric(self) -> None:
        """Test validation fails for unknown turn metric."""
        config = SystemConfig(
            default_turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.7},
                "custom:answer_correctness": {"threshold": 0.8},
            },
        )
        validator = DataValidator(system_config=config)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            turn_metrics=["unknown:metric"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert len(validator.validation_errors) > 0
        assert any(
            "Unknown turn metric" in error for error in validator.validation_errors
        )

    def test_validate_metrics_availability_unknown_conversation_metric(self) -> None:
        """Test validation fails for unknown conversation metric."""
        config = SystemConfig(
            default_conversation_metrics_metadata={
                "deepeval:conversation_completeness": {"threshold": 0.6},
            },
        )
        validator = DataValidator(system_config=config)

        turn = TurnData(turn_id="1", query="Query", response="Response")
        conv_data = EvaluationData(
            conversation_group_id="test_conv",
            turns=[turn],
            conversation_metrics=["unknown:conversation_metric"],
        )

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any(
            "Unknown conversation metric" in error
            for error in validator.validation_errors
        )

    def test_validate_metric_requirements_missing_response(self) -> None:
        """Test validation fails when required response field is missing."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response=None,  # Missing response
            turn_metrics=["ragas:faithfulness"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any("response" in error.lower() for error in validator.validation_errors)

    def test_validate_metric_requirements_missing_contexts(self) -> None:
        """Test validation fails when required contexts are missing."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            contexts=None,  # Missing contexts
            turn_metrics=["ragas:faithfulness"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any("contexts" in error.lower() for error in validator.validation_errors)

    def test_validate_metric_requirements_api_enabled_allows_missing_response(
        self,
    ) -> None:
        """Test that missing response is allowed when API is enabled."""
        config = SystemConfig(
            default_turn_metrics_metadata={
                "ragas:response_relevancy": {"threshold": 0.7},
            },
        )
        validator = DataValidator(api_enabled=True, system_config=config)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response=None,  # Will be populated by API
            turn_metrics=["ragas:response_relevancy"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        # Should pass because API will populate response
        assert result is True

    def test_validate_metric_requirements_expected_response_missing(self) -> None:
        """Test validation fails when expected_response is required but missing."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            contexts=["Context"],
            expected_response=None,  # Required for context_recall
            turn_metrics=["ragas:context_recall"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any(
            "expected_response" in error.lower()
            for error in validator.validation_errors
        )

    def test_validate_metric_requirements_tool_eval_missing_fields(self) -> None:
        """Test validation fails when tool_eval required fields are missing."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            tool_calls=None,  # Missing
            expected_tool_calls=None,  # Missing
            turn_metrics=["custom:tool_eval"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any(
            "tool_calls" in error.lower() for error in validator.validation_errors
        )

    def test_validate_metric_requirements_skip_script_when_api_disabled(self) -> None:
        """Test script metrics validation is skipped when API is disabled."""
        config = SystemConfig(
            default_turn_metrics_metadata={
                "script:action_eval": {"threshold": 0.5},
            },
        )
        validator = DataValidator(api_enabled=False, system_config=config)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            verify_script=None,  # Missing but should be ignored
            turn_metrics=["script:action_eval"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        # Should not validate script requirements when API disabled
        result = validator._validate_evaluation_data([conv_data])

        # Should pass because script validation is skipped
        assert result is True

    def test_load_evaluation_data_file_not_found(self) -> None:
        """Test loading non-existent file raises error."""
        validator = DataValidator()

        with pytest.raises(DataValidationError, match="file not found"):
            validator.load_evaluation_data("/nonexistent/file.yaml")

    def test_load_evaluation_data_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name

        try:
            validator = DataValidator()
            with pytest.raises(DataValidationError, match="Invalid YAML syntax"):
                validator.load_evaluation_data(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_evaluation_data_empty_file(self) -> None:
        """Test loading empty YAML file raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            validator = DataValidator()
            with pytest.raises(DataValidationError, match="Empty or invalid"):
                validator.load_evaluation_data(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_evaluation_data_not_list(self) -> None:
        """Test loading YAML with non-list root raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("conversation_group_id: test\n")
            temp_path = f.name

        try:
            validator = DataValidator()
            with pytest.raises(DataValidationError, match="must be a list"):
                validator.load_evaluation_data(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_evaluation_data_valid(self) -> None:
        """Test loading valid evaluation data file."""
        yaml_content = """
- conversation_group_id: test_conv
  turns:
    - turn_id: "1"
      query: "What is Python?"
      response: "Python is a programming language."
            """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            # Use empty SystemConfig so metric sets are empty (no false positives)
            config = SystemConfig()
            validator = DataValidator(api_enabled=False, system_config=config)
            data = validator.load_evaluation_data(temp_path)

            assert len(data) == 1
            assert data[0].conversation_group_id == "test_conv"
            assert len(data[0].turns) == 1
            assert data[0].turns[0].turn_id == "1"
        finally:
            Path(temp_path).unlink()

    def test_check_metric_requirements_missing_contexts(self) -> None:
        """Test validation fails for missing contexts when required."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response="Response",
            contexts=None,  # Missing contexts
            turn_metrics=["ragas:faithfulness"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False
        assert any("contexts" in error.lower() for error in validator.validation_errors)

    def test_check_metric_requirements_whitespace_only_string(self) -> None:
        """Test validation fails for whitespace-only required string."""
        validator = DataValidator(api_enabled=False)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response="   ",  # Whitespace only
            turn_metrics=["ragas:response_relevancy"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = validator._validate_evaluation_data([conv_data])

        assert result is False

    def test_validate_multiple_conversations(self) -> None:
        """Test validating multiple conversations."""
        validator = DataValidator(api_enabled=False)

        turn1 = TurnData(turn_id="1", query="Q1", response="R1")
        turn2 = TurnData(turn_id="1", query="Q2", response="R2")

        conv1 = EvaluationData(conversation_group_id="conv1", turns=[turn1])
        conv2 = EvaluationData(conversation_group_id="conv2", turns=[turn2])

        result = validator._validate_evaluation_data([conv1, conv2])

        assert result is True

    def test_validate_evaluation_data_accumulates_errors(self) -> None:
        """Test that validation accumulates multiple errors."""
        config = SystemConfig(
            default_turn_metrics_metadata={
                "ragas:faithfulness": {"threshold": 0.7},
            },
        )
        validator = DataValidator(api_enabled=False, system_config=config)

        # Create data with multiple validation issues
        turn1 = TurnData(
            turn_id="1",
            query="Q1",
            response=None,  # Missing for faithfulness
            turn_metrics=["ragas:faithfulness"],
        )
        turn2 = TurnData(
            turn_id="2",
            query="Q2",
            response="R2",
            turn_metrics=["unknown:metric"],  # Unknown metric
        )

        conv = EvaluationData(conversation_group_id="test", turns=[turn1, turn2])

        result = validator._validate_evaluation_data([conv])

        assert result is False
        # Should have errors for both issues
        assert len(validator.validation_errors) >= 2


class TestFilterByScope:
    """Unit test for filter by scope."""

    def test_filter_by_scope_no_filter(self) -> None:
        """Test no filtering when both tags and conv_ids are None."""
        validator = DataValidator()
        data = [
            EvaluationData(
                conversation_group_id="conv_1",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
            EvaluationData(
                conversation_group_id="conv_2",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
        ]
        result = validator._filter_by_scope(data)
        assert len(result) == 2

    def test_filter_by_scope_tags_only(self) -> None:
        """Test filtering by tags only."""
        validator = DataValidator()
        data = [
            EvaluationData(
                conversation_group_id="conv_1",
                tag="basic",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
            EvaluationData(
                conversation_group_id="conv_2",
                tag="advanced",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
            EvaluationData(
                conversation_group_id="conv_3",
                tag="basic",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
        ]
        result = validator._filter_by_scope(data, tags=["basic"])
        assert len(result) == 2
        assert all(c.tag == "basic" for c in result)

    def test_filter_by_scope_conv_ids_only(self) -> None:
        """Test filtering by conversation IDs only."""
        validator = DataValidator()
        data = [
            EvaluationData(
                conversation_group_id="conv_1",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
            EvaluationData(
                conversation_group_id="conv_2",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
            EvaluationData(
                conversation_group_id="conv_3",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
        ]
        result = validator._filter_by_scope(data, conv_ids=["conv_1", "conv_3"])
        assert len(result) == 2
        assert {c.conversation_group_id for c in result} == {"conv_1", "conv_3"}

    def test_filter_by_scope_tags_and_conv_ids(self) -> None:
        """Test filtering by both tags and conv_ids uses OR logic."""
        validator = DataValidator()
        data = [
            EvaluationData(
                conversation_group_id="conv_1",
                tag="basic",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
            EvaluationData(
                conversation_group_id="conv_2",
                tag="advanced",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
            EvaluationData(
                conversation_group_id="conv_3",
                tag="tools",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
        ]
        result = validator._filter_by_scope(data, tags=["basic"], conv_ids=["conv_3"])
        assert len(result) == 2  # conv_1 (basic tag) + conv_3 (by ID)

    def test_filter_by_scope_no_match_returns_empty(self) -> None:
        """Test filtering with no matching criteria returns empty list."""
        validator = DataValidator()
        data = [
            EvaluationData(
                conversation_group_id="conv_1",
                tag="basic",
                turns=[TurnData(turn_id="t1", query="Q", response="A")],
            ),
        ]
        result = validator._filter_by_scope(data, tags=["nonexistent"])
        assert len(result) == 0
