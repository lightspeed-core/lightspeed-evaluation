"""Tests for results manager."""

from unittest.mock import Mock, patch

import pytest

from lsc_agent_eval.core.agent_goal_eval.models import EvaluationResult
from lsc_agent_eval.core.agent_goal_eval.results import ResultsManager


class TestResultsManager:
    """Test ResultsManager."""

    def test_init(self):
        """Test ResultsManager initialization."""
        manager = ResultsManager("test_results/")

        assert manager.result_dir == "test_results/"

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_success(self, mock_dataframe, mock_to_csv, mock_mkdir):
        """Test successful results saving."""
        # Setup test data
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="What is Kubernetes?",
                response="Kubernetes is a container orchestration platform",
                eval_type="judge-llm",
                result="PASS",
            ),
            EvaluationResult(
                eval_id="test_002",
                query="Deploy nginx",
                response="kubectl create deployment nginx --image=nginx",
                eval_type="script",
                result="PASS",
            ),
        ]

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify directory creation
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Verify DataFrame was created with correct data
        mock_dataframe.assert_called_once()
        call_args = mock_dataframe.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["eval_id"] == "test_001"
        assert call_args[1]["eval_id"] == "test_002"

        # Verify to_csv was called
        mock_df_instance.to_csv.assert_called_once()

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_with_error(self, mock_dataframe, mock_to_csv, mock_mkdir):
        """Test results saving with error field."""
        # Setup test data with error
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="",
                eval_type="script",
                result="ERROR",
                error="Script execution failed",
            ),
        ]

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify DataFrame was created with error field
        mock_dataframe.assert_called_once()
        call_args = mock_dataframe.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["error"] == "Script execution failed"

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_empty_list(self, mock_dataframe, mock_to_csv, mock_mkdir):
        """Test saving empty results list."""
        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results([])

        # Verify DataFrame was created with empty list
        mock_dataframe.assert_called_once_with([])
        mock_df_instance.to_csv.assert_called_once()

    @patch("pathlib.Path.mkdir", side_effect=OSError("Permission denied"))
    def test_save_results_mkdir_error(self, mock_mkdir):
        """Test results saving with directory creation error."""
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="Test response",
                eval_type="judge-llm",
                result="PASS",
            ),
        ]

        manager = ResultsManager("test_results/")

        with pytest.raises(OSError, match="Permission denied"):
            manager.save_results(results)

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv", side_effect=IOError("File write error"))
    @patch("pandas.DataFrame")
    def test_save_results_file_error(self, mock_dataframe, mock_to_csv, mock_mkdir):
        """Test results saving with file write error."""
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="Test response",
                eval_type="judge-llm",
                result="PASS",
            ),
        ]

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance
        mock_df_instance.to_csv.side_effect = IOError("File write error")

        manager = ResultsManager("test_results/")

        with pytest.raises(IOError, match="File write error"):
            manager.save_results(results)

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    @patch("lsc_agent_eval.core.agent_goal_eval.results.datetime")
    def test_save_results_filename_generation(
        self, mock_datetime, mock_dataframe, mock_to_csv, mock_mkdir
    ):
        """Test CSV filename generation with timestamp."""
        # Setup mock datetime
        mock_datetime.now.return_value.strftime.return_value = "20240108_103000"

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="Test response",
                eval_type="judge-llm",
                result="PASS",
            ),
        ]

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify to_csv was called with correct path
        mock_df_instance.to_csv.assert_called_once()
        call_args = mock_df_instance.to_csv.call_args
        file_path = call_args[0][0]
        assert "agent_goal_eval_results_20240108_103000.csv" in str(file_path)

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_csv_parameters(self, mock_dataframe, mock_to_csv, mock_mkdir):
        """Test CSV parameters are correct."""
        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="Test response",
                eval_type="judge-llm",
                result="PASS",
            ),
        ]

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify to_csv was called with correct parameters
        mock_df_instance.to_csv.assert_called_once()
        call_args = mock_df_instance.to_csv.call_args
        assert not call_args[1]["index"]
        assert call_args[1]["encoding"] == "utf-8"

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_data_conversion(
        self, mock_dataframe, mock_to_csv, mock_mkdir
    ):
        """Test EvaluationResult to dict conversion."""
        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="Test response",
                eval_type="judge-llm",
                result="PASS",
                error=None,
            ),
        ]

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify DataFrame was created with correct data
        mock_dataframe.assert_called_once()
        call_args = mock_dataframe.call_args[0][0]
        expected_row = {
            "eval_id": "test_001",
            "query": "Test query",
            "response": "Test response",
            "eval_type": "judge-llm",
            "result": "PASS",
            "error": "",
        }
        assert call_args[0] == expected_row

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_multiple_results(
        self, mock_dataframe, mock_to_csv, mock_mkdir
    ):
        """Test saving multiple results."""
        # Setup test data
        results = [
            EvaluationResult("test_001", "query1", "response1", "judge-llm", "PASS"),
            EvaluationResult("test_002", "query2", "response2", "script", "FAIL"),
            EvaluationResult("test_003", "query3", "response3", "sub-string", "PASS"),
        ]

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify DataFrame was created with all results
        mock_dataframe.assert_called_once()
        call_args = mock_dataframe.call_args[0][0]
        assert len(call_args) == 3

        # Verify each result was converted correctly
        assert call_args[0]["eval_id"] == "test_001"
        assert call_args[1]["eval_id"] == "test_002"
        assert call_args[2]["eval_id"] == "test_003"

    def test_result_dir_with_trailing_slash(self):
        """Test result directory with trailing slash."""
        manager = ResultsManager("test_results/")
        assert manager.result_dir == "test_results/"

    def test_result_dir_without_trailing_slash(self):
        """Test result directory without trailing slash."""
        manager = ResultsManager("test_results")
        assert manager.result_dir == "test_results"

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_encoding(self, mock_dataframe, mock_to_csv, mock_mkdir):
        """Test CSV file is saved with UTF-8 encoding."""
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="What is Kubernetes?",
                response="Kubernetes is a container orchestration platform",
                eval_type="judge-llm",
                result="PASS",
            ),
        ]

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify to_csv was called with UTF-8 encoding
        call_args = mock_df_instance.to_csv.call_args
        assert call_args[1]["encoding"] == "utf-8"

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_no_index(self, mock_dataframe, mock_to_csv, mock_mkdir):
        """Test CSV file index handling."""
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="Test response",
                eval_type="judge-llm",
                result="PASS",
            ),
        ]

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify to_csv was called with index=False
        call_args = mock_df_instance.to_csv.call_args
        assert not call_args[1]["index"]

    @patch("pathlib.Path.mkdir")
    @patch("pandas.DataFrame.to_csv")
    @patch("pandas.DataFrame")
    def test_save_results_none_error_handling(
        self, mock_dataframe, mock_to_csv, mock_mkdir
    ):
        """Test handling of None error values."""
        results = [
            EvaluationResult(
                eval_id="test_001",
                query="Test query",
                response="Test response",
                eval_type="judge-llm",
                result="PASS",
                error=None,
            ),
        ]

        # Setup mocks
        mock_df_instance = Mock()
        mock_dataframe.return_value = mock_df_instance

        # Run test
        manager = ResultsManager("test_results/")
        manager.save_results(results)

        # Verify None error is converted to empty string
        mock_dataframe.assert_called_once()
        call_args = mock_dataframe.call_args[0][0]
        assert call_args[0]["error"] == ""

    def test_get_output_dir(self):
        """Test get_output_dir method."""
        manager = ResultsManager("test_results/")
        output_dir = manager.get_output_dir()
        assert output_dir == str(manager.result_path)
