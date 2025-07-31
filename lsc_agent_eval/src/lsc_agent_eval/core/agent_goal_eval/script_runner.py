"""Script execution module for evaluation."""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from ..utils.exceptions import ScriptExecutionError

logger = logging.getLogger(__name__)


class ScriptRunner:
    """Executes scripts for evaluation purposes."""

    def __init__(self, kubeconfig: Optional[str] = None):
        """Initialize script runner."""
        self.kubeconfig = kubeconfig

    def get_environment(self) -> dict:
        """Get environment variables for script execution."""
        env = os.environ.copy()
        if self.kubeconfig:
            env["KUBECONFIG"] = self.kubeconfig
        return env

    def run_script(self, script_path: Union[str, Path]) -> bool:
        """Execute a script and return success status."""
        if isinstance(script_path, str):
            script_path = Path(script_path)
        script_path = script_path.resolve()

        if not script_path.exists():
            raise ScriptExecutionError(f"Script not found: {script_path}")

        if not script_path.is_file():
            raise ScriptExecutionError(f"Script path is not a file: {script_path}")

        try:
            # Setup environment
            env = self.get_environment()

            # Make script executable
            script_path.chmod(0o755)

            # Run script
            logger.debug("Running script: %s", script_path)

            result = subprocess.run(
                [str(script_path)],
                text=True,
                capture_output=True,
                env=env,
                timeout=300,  # 5 minute timeout
                check=False,
            )

            # Log output
            if result.stdout:
                logger.debug("Script stdout: %s", result.stdout)
            if result.stderr:
                logger.warning("Script stderr: %s", result.stderr)

            return result.returncode == 0

        except subprocess.TimeoutExpired as e:
            raise ScriptExecutionError(f"Script timeout: {script_path}") from e
        except subprocess.SubprocessError as e:
            raise ScriptExecutionError(
                f"Error running script {script_path}: {e}"
            ) from e
        except Exception as e:
            raise ScriptExecutionError(
                f"Unexpected error running script {script_path}: {e}"
            ) from e
