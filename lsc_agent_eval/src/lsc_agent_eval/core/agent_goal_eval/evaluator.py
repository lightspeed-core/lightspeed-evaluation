"""Evaluation runner that orchestrates different evaluation types."""

import logging
from typing import Optional

from ..utils.api_client import AgentHttpClient
from ..utils.exceptions import AgentAPIError, JudgeModelError, ScriptExecutionError
from ..utils.judge import JudgeModelManager
from ..utils.prompt import ANSWER_CORRECTNESS_PROMPT
from .models import EvaluationDataConfig, EvaluationResult
from .script_runner import ScriptRunner

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates different types of evaluations."""

    def __init__(
        self,
        agent_client: AgentHttpClient,
        judge_manager: Optional[JudgeModelManager] = None,
        kubeconfig: Optional[str] = None,
    ):
        """Initialize evaluation runner."""
        self.agent_client = agent_client
        self.judge_manager = judge_manager
        self.kubeconfig = kubeconfig

    def run_evaluation(
        self, data_config: EvaluationDataConfig, agent_provider: str, agent_model: str
    ) -> EvaluationResult:
        """Run a single evaluation based on configuration."""
        try:
            # Execute setup script if provided
            if data_config.eval_setup_script:
                try:
                    script_runner = ScriptRunner(kubeconfig=self.kubeconfig)
                    success = script_runner.run_script(data_config.eval_setup_script)
                    if not success:
                        raise ScriptExecutionError(
                            "Setup script returned non-zero exit code"
                        )
                    logger.debug(
                        "Setup script executed successfully for %s", data_config.eval_id
                    )
                except ScriptExecutionError as e:
                    logger.error(
                        "Setup script failed for %s: %s", data_config.eval_id, e
                    )
                    return EvaluationResult(
                        eval_id=data_config.eval_id,
                        query=data_config.eval_query,
                        response="",
                        eval_type=data_config.eval_type,
                        result="ERROR",
                        error=f"Setup script failed: {e}",
                    )

            response = self.agent_client.query_agent(
                data_config.eval_query, agent_provider, agent_model
            )

            # Evaluate response based on type
            success = self._evaluate_response(data_config, response)

            # Execute cleanup script if provided
            if data_config.eval_cleanup_script:
                try:
                    cleanup_runner = ScriptRunner(kubeconfig=self.kubeconfig)
                    cleanup_success = cleanup_runner.run_script(
                        data_config.eval_cleanup_script
                    )
                    if cleanup_success:
                        logger.debug(
                            "Cleanup script executed successfully for %s",
                            data_config.eval_id,
                        )
                    else:
                        logger.warning(
                            "Cleanup script failed for %s", data_config.eval_id
                        )
                except ScriptExecutionError as e:
                    logger.warning(
                        "Cleanup script failed for %s: %s", data_config.eval_id, e
                    )

            return EvaluationResult(
                eval_id=data_config.eval_id,
                query=data_config.eval_query,
                response=response,
                eval_type=data_config.eval_type,
                result="PASS" if success else "FAIL",
            )

        except (AgentAPIError, ScriptExecutionError, JudgeModelError) as e:
            logger.error("Evaluation failed for %s: %s", data_config.eval_id, e)
            return EvaluationResult(
                eval_id=data_config.eval_id,
                query=data_config.eval_query,
                response="",
                eval_type=data_config.eval_type,
                result="ERROR",
                error=str(e),
            )

    def _evaluate_response(
        self, data_config: EvaluationDataConfig, response: str
    ) -> bool:
        """Evaluate response based on configuration type."""
        match data_config.eval_type:
            case "script":
                return self._evaluate_script(data_config)
            case "sub-string":
                return self._evaluate_substring(data_config, response)
            case "judge-llm":
                return self._evaluate_judge_llm(data_config, response)
            case _:
                logger.error("Unknown evaluation type: %s", data_config.eval_type)
                return False

    def _evaluate_script(self, data_config: EvaluationDataConfig) -> bool:
        """Evaluate using script execution."""
        if not data_config.eval_verify_script:
            logger.error("No verify script provided for script evaluation")
            return False

        try:
            script_runner = ScriptRunner(kubeconfig=self.kubeconfig)
            return script_runner.run_script(data_config.eval_verify_script)
        except ScriptExecutionError as e:
            logger.error("Script evaluation failed: %s", e)
            return False

    def _evaluate_substring(
        self, data_config: EvaluationDataConfig, response: str
    ) -> bool:
        """Evaluate using substring matching."""
        if not data_config.expected_keywords:
            return False

        response_lower = response.lower()
        return any(
            keyword.lower() in response_lower
            for keyword in data_config.expected_keywords
        )

    def _extract_numeric_result(self, response: Optional[str]) -> int:
        """Extract numeric result from judge response."""
        # Look for 1 or 0 in the response
        if response:
            response = response.strip()

        if response not in ["0", "1"]:
            raise JudgeModelError(
                "Invalid response from the judge model. "
                f"Expected value either 0/1. Actual value: {response}"
            )

        return int(response)

    def _evaluate_judge_llm(
        self, data_config: EvaluationDataConfig, response: str
    ) -> bool:
        """Evaluate using judge LLM."""
        if not self.judge_manager:
            logger.error("Judge model manager not available for judge-llm evaluation")
            return False

        if not data_config.expected_response:
            logger.error("Expected response not provided for judge-llm evaluation")
            return False

        # Format prompt with parameters
        prompt = ANSWER_CORRECTNESS_PROMPT.format(
            question=data_config.eval_query,
            answer=data_config.expected_response,
            response=response,
        )
        judge_resp = self.judge_manager.evaluate_response(prompt)

        # Extract numeric result (looking for 1 or 0)
        result = self._extract_numeric_result(judge_resp)
        return result == 1

    def get_judge_manager(self) -> Optional[JudgeModelManager]:
        """Get the judge model manager."""
        return self.judge_manager
