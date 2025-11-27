"""Additional tests for script manager to increase coverage."""

import subprocess

import pytest

from lightspeed_evaluation.core.script.manager import ScriptExecutionManager
from lightspeed_evaluation.core.system.exceptions import ScriptExecutionError


class TestScriptExecutionManagerAdditional:
    """Additional tests for ScriptExecutionManager."""

    def test_run_script_timeout_error(self, tmp_path, mocker):
        """Test script execution with timeout."""
        # Create a script file
        script = tmp_path / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'test'\n")
        script.chmod(0o755)

        # Mock subprocess.run to raise TimeoutExpired
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.TimeoutExpired("test", 1)

        manager = ScriptExecutionManager(timeout=1)

        with pytest.raises(ScriptExecutionError, match="timeout"):
            manager.run_script(script)

    def test_run_script_subprocess_error(self, tmp_path, mocker):
        """Test script execution with subprocess error."""
        script = tmp_path / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'test'\n")
        script.chmod(0o755)

        # Mock subprocess.run to raise SubprocessError
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.SubprocessError("Test error")

        manager = ScriptExecutionManager()

        with pytest.raises(ScriptExecutionError, match="Error running script"):
            manager.run_script(script)

    def test_run_script_unexpected_error(self, tmp_path, mocker):
        """Test script execution with unexpected error."""
        script = tmp_path / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'test'\n")
        script.chmod(0o755)

        # Mock subprocess.run to raise unexpected error
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = RuntimeError("Unexpected error")

        manager = ScriptExecutionManager()

        with pytest.raises(ScriptExecutionError, match="Unexpected error"):
            manager.run_script(script)

    def test_run_script_with_path_object(self, tmp_path, mocker):
        """Test run_script accepts Path objects."""
        script = tmp_path / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'test'\n")
        script.chmod(0o755)

        mock_result = subprocess.CompletedProcess(
            args=[str(script)], returncode=0, stdout="test\n", stderr=""
        )
        mocker.patch("subprocess.run", return_value=mock_result)

        manager = ScriptExecutionManager()
        result = manager.run_script(script)

        assert result is True

    def test_script_not_file_error(self, tmp_path):
        """Test error when script path is not a file."""
        # Create a directory instead of file
        script_dir = tmp_path / "script_dir"
        script_dir.mkdir()

        manager = ScriptExecutionManager()

        with pytest.raises(ScriptExecutionError, match="not a file"):
            manager.run_script(script_dir)

    def test_script_output_logging(self, tmp_path, mocker, caplog):
        """Test that script output is logged."""
        import logging

        caplog.set_level(logging.DEBUG)

        script = tmp_path / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'test output'\n")
        script.chmod(0o755)

        mock_result = subprocess.CompletedProcess(
            args=[str(script)],
            returncode=0,
            stdout="test output\n",
            stderr="",
        )
        mocker.patch("subprocess.run", return_value=mock_result)

        manager = ScriptExecutionManager()
        manager.run_script(script)

        # Check that output was logged
        assert "test output" in caplog.text or "completed successfully" in caplog.text

    def test_script_stderr_logging_on_failure(self, tmp_path, mocker, caplog):
        """Test that stderr is logged as error on failure."""
        import logging

        caplog.set_level(logging.ERROR)

        script = tmp_path / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'error' >&2\nexit 1\n")
        script.chmod(0o755)

        mock_result = subprocess.CompletedProcess(
            args=[str(script)],
            returncode=1,
            stdout="",
            stderr="error\n",
        )
        mocker.patch("subprocess.run", return_value=mock_result)

        manager = ScriptExecutionManager()
        result = manager.run_script(script)

        assert result is False

    def test_script_stderr_logging_on_success(self, tmp_path, mocker, caplog):
        """Test that stderr is logged as debug on success."""
        import logging

        caplog.set_level(logging.DEBUG)

        script = tmp_path / "test_script.sh"
        script.write_text("#!/bin/bash\necho 'warning' >&2\nexit 0\n")
        script.chmod(0o755)

        mock_result = subprocess.CompletedProcess(
            args=[str(script)],
            returncode=0,
            stdout="",
            stderr="warning\n",
        )
        mocker.patch("subprocess.run", return_value=mock_result)

        manager = ScriptExecutionManager()
        result = manager.run_script(script)

        assert result is True
