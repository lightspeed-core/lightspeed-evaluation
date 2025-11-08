"""Unit tests for core script manager module."""

import tempfile
from pathlib import Path

import pytest

from lightspeed_evaluation.core.script.manager import ScriptExecutionManager
from lightspeed_evaluation.core.system.exceptions import ScriptExecutionError


class TestScriptExecutionManager:
    """Unit tests for ScriptExecutionManager."""

    def test_run_script_success(self):
        """Test running a successful script."""
        # Create a simple script that exits successfully
        script_content = "#!/bin/bash\nexit 0\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)

        try:
            # Make script executable
            script_path.chmod(0o755)

            manager = ScriptExecutionManager()
            result = manager.run_script(script_path)

            assert result is True
        finally:
            script_path.unlink()

    def test_run_script_failure(self):
        """Test running a script that fails."""
        # Create a script that exits with error code
        script_content = "#!/bin/bash\nexit 1\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)

        try:
            script_path.chmod(0o755)

            manager = ScriptExecutionManager()
            result = manager.run_script(script_path)

            assert result is False
        finally:
            script_path.unlink()

    def test_run_script_not_found(self):
        """Test running non-existent script raises error."""
        manager = ScriptExecutionManager()

        with pytest.raises(ScriptExecutionError, match="not found"):
            manager.run_script("/nonexistent/script.sh")

    def test_run_script_not_executable(self):
        """Test running non-executable file raises error."""
        script_content = "#!/bin/bash\nexit 0\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)

        try:
            # Don't make it executable
            script_path.chmod(0o644)

            manager = ScriptExecutionManager()

            with pytest.raises(ScriptExecutionError, match="not executable"):
                manager.run_script(script_path)
        finally:
            script_path.unlink()

    def test_run_script_not_a_file(self):
        """Test running a directory raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ScriptExecutionManager()

            with pytest.raises(ScriptExecutionError, match="not a file"):
                manager.run_script(tmpdir)

    def test_run_script_with_output(self):
        """Test script with stdout output."""
        script_content = '#!/bin/bash\necho "Test output"\nexit 0\n'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)

        try:
            script_path.chmod(0o755)

            manager = ScriptExecutionManager()
            result = manager.run_script(script_path)

            assert result is True
        finally:
            script_path.unlink()

    def test_run_script_with_stderr(self):
        """Test script with stderr output."""
        script_content = '#!/bin/bash\necho "Error message" >&2\nexit 1\n'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)

        try:
            script_path.chmod(0o755)

            manager = ScriptExecutionManager()
            result = manager.run_script(script_path)

            assert result is False
        finally:
            script_path.unlink()

    def test_run_script_accepts_string_path(self):
        """Test that run_script accepts string path."""
        script_content = "#!/bin/bash\nexit 0\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            Path(script_path).chmod(0o755)

            manager = ScriptExecutionManager()
            result = manager.run_script(script_path)  # Pass as string

            assert result is True
        finally:
            Path(script_path).unlink()

    def test_run_script_resolves_relative_path(self):
        """Test that relative paths are resolved."""
        script_content = "#!/bin/bash\nexit 0\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "test_script.sh"
            script_path.write_text(script_content)
            script_path.chmod(0o755)

            # Use relative path
            import os

            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                manager = ScriptExecutionManager()
                result = manager.run_script("./test_script.sh")
                assert result is True
            finally:
                os.chdir(original_cwd)

    def test_run_script_timeout(self):
        """Test script timeout raises error."""
        # Create a script that sleeps
        script_content = "#!/bin/bash\nsleep 10\nexit 0\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = Path(f.name)

        try:
            script_path.chmod(0o755)

            # Use very short timeout
            manager = ScriptExecutionManager(timeout=1)

            with pytest.raises(ScriptExecutionError, match="timeout"):
                manager.run_script(script_path)
        finally:
            script_path.unlink()
