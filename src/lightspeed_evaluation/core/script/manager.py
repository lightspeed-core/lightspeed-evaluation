"""Script execution manager for evaluation framework."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Union

from ..system.exceptions import ScriptExecutionError

logger = logging.getLogger(__name__)


class ScriptExecutionManager:  # pylint: disable=too-few-public-methods
    """Executes scripts for evaluation purposes."""

    def __init__(self, timeout: int = 300):
        """Initialize script execution manager.

        Args:
            timeout: Script execution timeout in seconds (default: 5 minutes)

        Note:
            KUBECONFIG environment variable will be used if set.
        """
        self.timeout = timeout

    def run_script(self, script_path: Union[str, Path]) -> bool:
        """Execute a script and return success status.

        Args:
            script_path: Path to the script to execute (should be absolute path from validation)

        Returns:
            True if script executed successfully (exit code 0), False otherwise

        Raises:
            ScriptExecutionError: If script execution fails due to system errors
        """
        script_path = self._prepare_script_path(script_path)
        self._validate_script_path(script_path)

        try:
            logger.debug("Running script: %s", script_path)
            result = self._execute_script(script_path)
            return self._process_result(result, script_path)
        except subprocess.TimeoutExpired as e:
            raise ScriptExecutionError(
                f"Script timeout after {self.timeout}s: {script_path}", str(script_path)
            ) from e
        except subprocess.SubprocessError as e:
            raise ScriptExecutionError(
                f"Error running script {script_path}: {e}", str(script_path)
            ) from e
        except Exception as e:
            raise ScriptExecutionError(
                f"Unexpected error running script {script_path}: {e}", str(script_path)
            ) from e

    def _prepare_script_path(self, script_path: Union[str, Path]) -> Path:
        """Prepare and resolve script path."""
        if isinstance(script_path, str):
            script_path = Path(script_path)
        return script_path.resolve()

    def _validate_script_path(self, script_path: Path) -> None:
        """Validate script path exists and is executable."""
        if not script_path.exists():
            raise ScriptExecutionError(
                f"Script not found: {script_path}", str(script_path)
            )

        if not script_path.is_file():
            raise ScriptExecutionError(
                f"Script path is not a file: {script_path}", str(script_path)
            )

        if not os.access(script_path, os.X_OK):
            raise ScriptExecutionError(
                f"Script is not executable: {script_path}", str(script_path)
            )

    def _execute_script(self, script_path: Path) -> subprocess.CompletedProcess:
        """Execute the script and return the result."""
        env = os.environ.copy()
        return subprocess.run(
            [str(script_path)],
            text=True,
            capture_output=True,
            env=env,
            cwd=script_path.parent,
            timeout=self.timeout,
            check=False,
        )

    def _process_result(
        self, result: subprocess.CompletedProcess, script_path: Path
    ) -> bool:
        """Process script execution result and log output."""
        # Log output
        if result.stdout:
            logger.debug("Script stdout: %s", result.stdout.strip())

        if result.stderr:
            log_func = logger.error if result.returncode != 0 else logger.debug
            log_func("Script stderr: %s", result.stderr.strip())

        success = result.returncode == 0
        log_func = logger.debug if success else logger.warning
        message = (
            "completed successfully"
            if success
            else f"failed with exit code {result.returncode}"
        )
        log_func("Script %s %s", script_path, message)

        return success
