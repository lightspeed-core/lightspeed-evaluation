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
            script_path: Path to the script to execute

        Returns:
            True if script executed successfully (exit code 0), False otherwise

        Raises:
            ScriptExecutionError: If script execution fails due to system errors
        """
        if isinstance(script_path, str):
            script_path = Path(script_path)

        script_path = script_path.resolve()

        if not script_path.exists():
            raise ScriptExecutionError(
                f"Script not found: {script_path}", str(script_path)
            )

        if not script_path.is_file():
            raise ScriptExecutionError(
                f"Script path is not a file: {script_path}", str(script_path)
            )

        try:
            # Use current environment (includes KUBECONFIG if set)
            env = os.environ.copy()

            # Run script
            logger.debug("Running script: %s", script_path)

            result = subprocess.run(
                [str(script_path)],
                text=True,
                capture_output=True,
                env=env,
                timeout=self.timeout,
                check=False,
            )

            # Log output
            if result.stdout:
                logger.debug("Script stdout: %s", result.stdout.strip())
            if result.stderr:
                if result.returncode != 0:
                    logger.error("Script stderr: %s", result.stderr.strip())
                else:
                    logger.debug("Script stderr: %s", result.stderr.strip())

            success = result.returncode == 0
            if not success:
                logger.warning(
                    "Script %s failed with exit code %d", script_path, result.returncode
                )
            else:
                logger.debug("Script %s completed successfully", script_path)

            return success

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
