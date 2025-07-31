"""Evaluation runner that orchestrates different evaluation types."""

import logging
from typing import TYPE_CHECKING, Optional

from ..utils.exceptions import AgentAPIError, JudgeModelError, ScriptExecutionError
from ..utils.prompt import ANSWER_CORRECTNESS_PROMPT
from .utils import create_error_result, create_success_result

if TYPE_CHECKING:
    from ..utils.api_client import AgentHttpClient
    from ..utils.judge import JudgeModelManager
    from .models import EvaluationDataConfig, EvaluationResult
    from .script_runner import ScriptRunner

logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Orchestrates different types of evaluations."""

    def __init__(
        self,
        agent_client: "AgentHttpClient",
        script_runner: "ScriptRunner",
        judge_manager: Optional["JudgeModelManager"] = None,
    ):
        """Initialize evaluation runner."""
        self.agent_client = agent_client
        self.judge_manager = judge_manager
        self.script_runner = script_runner

    def run_evaluation(
        self,
        data_config: "EvaluationDataConfig",
        agent_provider: str,
        agent_model: str,
        conversation_id: Optional[str] = None,
    ) -> "EvaluationResult":
        """Run a single evaluation based on configuration."""
        try:
            # Query the agent
            api_input = {
                "query": data_config.eval_query,
                "provider": agent_provider,
                "model": agent_model,
                "conversation_id": conversation_id,
            }

            response, conversation_id = self.agent_client.query_agent(api_input)

            # Evaluate agent action based on eval type
            success = self._evaluate_agent_action(data_config, response)

            return create_success_result(
                data_config, response, success, conversation_id
            )

        except (AgentAPIError, ScriptExecutionError, JudgeModelError) as e:
            logger.error("Evaluation failed for %s: %s", data_config.eval_id, e)
            return create_error_result(data_config, str(e), conversation_id)

    def _evaluate_agent_action(
        self, data_config: "EvaluationDataConfig", response: str
    ) -> bool:
        """Evaluate agent action based on configuration type."""
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

    def _evaluate_script(self, data_config: "EvaluationDataConfig") -> bool:
        """Evaluate using script execution."""
        if not data_config.eval_verify_script:
            logger.error("No verify script provided for script evaluation")
            return False

        return self.script_runner.run_script(data_config.eval_verify_script)

    def _evaluate_substring(
        self, data_config: "EvaluationDataConfig", response: str
    ) -> bool:
        """Evaluate using substring matching."""
        if not data_config.expected_keywords:
            return False

        response_lower = response.lower()
        # All keywords must be present for evaluation to pass
        for keyword in data_config.expected_keywords:
            if keyword.lower() not in response_lower:
                return False
        return True

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
        self, data_config: "EvaluationDataConfig", response: str
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

    def get_judge_manager(self) -> Optional["JudgeModelManager"]:
        """Get the judge model manager."""
        return self.judge_manager
