"""Script-based evaluation metrics."""

import logging
from pathlib import Path
from typing import Any, Optional, Union

from ..models import EvaluationScope
from ..script import ScriptExecutionError, ScriptExecutionManager

logger = logging.getLogger(__name__)


class ScriptEvalMetrics:  # pylint: disable=too-few-public-methods
    """Script-based evaluation metrics."""

    def __init__(
        self, script_manager: ScriptExecutionManager, api_enabled: bool = True
    ):
        """Initialize Script Evaluation."""
        self.script_manager = script_manager
        self.api_enabled = api_enabled

    def evaluate(
        self, metric_name: str, _: Any, evaluation_scope: EvaluationScope
    ) -> tuple[Optional[float], str]:
        """Evaluate script-based metric."""
        if evaluation_scope.is_conversation:
            return None, "Script evaluation is only supported for turn-level metrics"

        if not evaluation_scope.turn_data:
            return None, "Turn data is required for script evaluation"

        turn_data = evaluation_scope.turn_data

        if metric_name == "action_eval":
            return self._evaluate_verify_script(turn_data.verify_script)

        return None, f"Unsupported script metric: {metric_name}"

    def _evaluate_verify_script(
        self, script_path: Optional[Union[str, Path]]
    ) -> tuple[Optional[float], str]:
        """Evaluate verify script."""
        if not script_path:
            return None, "No verify script provided"

        # Skip script execution if API is disabled
        if not self.api_enabled:
            logger.debug("Skipping verify script (API disabled): %s", script_path)
            return None, f"Verify script skipped (API disabled): {script_path}"

        try:
            success = self.script_manager.run_script(script_path)
            if success:
                return 1.0, f"Verify script passed: {script_path}"

            return 0.0, f"Verify script failed: {script_path}"
        except ScriptExecutionError as e:
            logger.error("Script execution error: %s", e)
            return None, f"Script execution error: {e}"
