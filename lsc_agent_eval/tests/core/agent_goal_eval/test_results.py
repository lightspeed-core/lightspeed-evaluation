"""Tests for results manager."""

import json
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pandas as pd
import pytest

from lsc_agent_eval.core.agent_goal_eval.models import EvaluationResult, EvaluationStats
from lsc_agent_eval.core.agent_goal_eval.results import ResultsManager
from lsc_agent_eval.core.utils.exceptions import AgentEvaluationError


class TestResultsManager:
    """Test ResultsManager."""

    @pytest.fixture
    def sample_results(self):
        """Sample evaluation results."""
        return [
            EvaluationResult(
                eval_id="test_001",
                query="What is Kubernetes?",
                response="Kubernetes is a container orchestration platform",
                eval_type="response_eval:accuracy",
                result="PASS",
                conversation_group="conv1",
                conversation_id="conv-id-123",
            ),
            EvaluationResult(
                eval_id="test_002",
                query="Deploy nginx",
                response="oc create deployment nginx --image=nginx",
                eval_type="action_eval",
                result="FAIL",
                conversation_group="conv1",
                conversation_id="conv-id-123",
            ),
            EvaluationResult(
                eval_id="test_003",
                query="List pods",
                response="pod1, pod2",
                eval_type="response_eval:sub-string",
                result="PASS",
                conversation_group="conv2",
                conversation_id="conv-id-456",
            ),
        ]

    @pytest.fixture
    def empty_results(self):
        """Empty results list."""
        return []

    def test_init(self, sample_results):
        """Test ResultsManager initialization."""
        manager = ResultsManager(sample_results)

        assert manager.results == sample_results
        assert isinstance(manager.results_stats, EvaluationStats)
        assert manager.results_stats.total_evaluations == 3
        assert manager.results_stats.passed == 2
        assert manager.results_stats.failed == 1

    def test_init_empty_results(self, empty_results):
        """Test ResultsManager initialization with empty results."""
        manager = ResultsManager(empty_results)

        assert manager.results == []
        assert manager.results_stats.total_evaluations == 0

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_results_success(
        self, mock_file_open, mock_to_csv, mock_mkdir, sample_results
    ):
        """Test successful results saving."""
        manager = ResultsManager(sample_results)

        manager.save_results("test_results/")

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify CSV saving
        mock_to_csv.assert_called_once()

        # Verify JSON saving
        mock_file_open.assert_called()

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv", side_effect=Exception("CSV error"))
    def test_save_results_csv_error(self, mock_to_csv, mock_mkdir, sample_results):
        """Test results saving with CSV error."""
        manager = ResultsManager(sample_results)

        with pytest.raises(AgentEvaluationError, match="Failed to save results"):
            manager.save_results("test_results/")

    @patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied"))
    def test_save_results_mkdir_error(self, mock_mkdir, sample_results):
        """Test results saving with directory creation error."""
        manager = ResultsManager(sample_results)

        with pytest.raises(AgentEvaluationError, match="Failed to save results"):
            manager.save_results("test_results/")

    def test_csv_data_conversion(self, sample_results):
        """Test CSV data conversion."""
        manager = ResultsManager(sample_results)

        data = []
        for result in manager.results:
            data.append(
                {
                    "conversation_group": result.conversation_group,
                    "conversation_id": result.conversation_id,
                    "eval_id": result.eval_id,
                    "result": result.result,
                    "eval_type": result.eval_type,
                    "query": result.query,
                    "response": result.response,
                    "error": result.error,
                }
            )

        assert len(data) == 3
        assert data[0]["eval_id"] == "test_001"
        assert data[0]["result"] == "PASS"
        assert data[1]["result"] == "FAIL"
        assert data[2]["eval_type"] == "response_eval:sub-string"

    def test_get_results_stats(self, sample_results):
        """Test get results stats method."""
        manager = ResultsManager(sample_results)
        stats = manager.get_results_stats()

        assert isinstance(stats, EvaluationStats)
        assert stats.total_evaluations == 3
        assert stats.total_conversations == 2
        assert stats.passed == 2
        assert stats.failed == 1
        assert stats.errored == 0
        assert round(stats.success_rate, 2) == round(2 / 3 * 100, 2)

        # Check conversation breakdown
        assert "conv1" in stats.by_conversation
        assert "conv2" in stats.by_conversation
        assert stats.by_conversation["conv1"]["total"] == 2
        assert stats.by_conversation["conv2"]["total"] == 1

        # Check eval type breakdown
        assert "response_eval:accuracy" in stats.by_eval_type
        assert "action_eval" in stats.by_eval_type
        assert "response_eval:sub-string" in stats.by_eval_type

    def test_results_with_errors(self):
        """Test results with error conditions."""
        results = [
            EvaluationResult(
                eval_id="test_error",
                query="Failing query",
                response="",
                eval_type="response_eval:accuracy",
                result="ERROR",
                error="API connection failed",
                conversation_group="test_conv",
                conversation_id="conv-id-789",
            ),
        ]

        manager = ResultsManager(results)
        stats = manager.get_results_stats()

        assert stats.total_evaluations == 1
        assert stats.passed == 0
        assert stats.failed == 0
        assert stats.errored == 1
        assert stats.success_rate == 0.0

    def test_results_mixed_types(self):
        """Test results with mixed evaluation types."""
        results = [
            EvaluationResult(
                eval_id="judge_test",
                query="Judge query",
                response="Judge response",
                eval_type="response_eval:accuracy",
                result="PASS",
                conversation_group="mixed_conv",
                conversation_id="conv-id-mixed",
            ),
            EvaluationResult(
                eval_id="script_test",
                query="Script query",
                response="Script response",
                eval_type="action_eval",
                result="FAIL",
                conversation_group="mixed_conv",
                conversation_id="conv-id-mixed",
            ),
            EvaluationResult(
                eval_id="substring_test",
                query="Substring query",
                response="Substring response",
                eval_type="response_eval:sub-string",
                result="PASS",
                conversation_group="mixed_conv",
                conversation_id="conv-id-mixed",
            ),
        ]

        manager = ResultsManager(results)
        stats = manager.get_results_stats()

        assert stats.total_evaluations == 3
        assert stats.total_conversations == 1
        assert len(stats.by_eval_type) == 3
        assert stats.by_eval_type["response_eval:accuracy"]["passed"] == 1
        assert stats.by_eval_type["action_eval"]["failed"] == 1
        assert stats.by_eval_type["response_eval:sub-string"]["passed"] == 1

    def test_json_statistics_structure(self, sample_results):
        """Test JSON statistics structure."""
        manager = ResultsManager(sample_results)
        stats = manager.get_results_stats()

        # Convert to dict as would be saved to JSON
        stats_dict = stats.model_dump()

        assert "total_evaluations" in stats_dict
        assert "total_conversations" in stats_dict
        assert "passed" in stats_dict
        assert "failed" in stats_dict
        assert "errored" in stats_dict
        assert "success_rate" in stats_dict
        assert "by_conversation" in stats_dict
        assert "by_eval_type" in stats_dict

        # Verify structure of nested stats
        assert isinstance(stats_dict["by_conversation"], dict)
        assert isinstance(stats_dict["by_eval_type"], dict)

    def test_filename_generation_format(self, sample_results):
        """Test that filename generation follows expected format."""
        manager = ResultsManager(sample_results)

        with patch(
            "lsc_agent_eval.core.agent_goal_eval.results.datetime"
        ) as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            with (
                patch.object(manager, "_save_csv_results"),
                patch.object(manager, "_save_json_summary"),
                patch("pathlib.Path.mkdir"),
            ):

                manager.save_results("test_results/")

                # Verify the filename format is called correctly
                mock_datetime.now.assert_called_once()
                mock_datetime.now.return_value.strftime.assert_called_once_with(
                    "%Y%m%d_%H%M%S"
                )

    def test_integration_with_real_files(self, sample_results):
        """Integration test with real temporary files."""
        manager = ResultsManager(sample_results)

        with tempfile.TemporaryDirectory() as temp_dir:
            manager.save_results(temp_dir)

            # Check that files were created
            result_files = list(Path(temp_dir).glob("agent_goal_eval_results_*.csv"))
            summary_files = list(Path(temp_dir).glob("agent_goal_eval_summary_*.json"))

            assert len(result_files) == 1
            assert len(summary_files) == 1

            # Verify CSV content
            csv_data = pd.read_csv(result_files[0])
            assert len(csv_data) == 3
            assert "eval_id" in csv_data.columns
            assert "result" in csv_data.columns
            assert "conversation_group" in csv_data.columns

            # Verify JSON content
            with open(summary_files[0], "r") as f:
                json_data = json.load(f)

            assert json_data["summary"]["total_evaluations"] == 3
            assert json_data["summary"]["passed"] == 2
            assert "by_conversation" in json_data
            assert "by_eval_type" in json_data
