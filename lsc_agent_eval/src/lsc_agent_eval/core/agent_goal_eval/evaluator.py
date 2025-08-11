"""Evaluation runner that orchestrates different evaluation types."""

import logging
from typing import TYPE_CHECKING, Optional

from ..utils.exceptions import AgentAPIError, JudgeModelError, ScriptExecutionError
from ..utils.prompt import ANSWER_CORRECTNESS_PROMPT
from .tool_call_eval import compare_tool_calls
from .utils import create_evaluation_results

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

    def run_evaluation(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        data_config: "EvaluationDataConfig",
        agent_provider: str,
        agent_model: str,
        conversation_id: Optional[str] = None,
        endpoint_type: str = "streaming",
    ) -> list["EvaluationResult"]:
        """Run multiple evaluations based on configuration."""
        try:
            # Query the agent once
            api_input = {
                "query": data_config.eval_query,
                "provider": agent_provider,
                "model": agent_model,
                "conversation_id": conversation_id,
            }

            if endpoint_type == "streaming":
                agent_response = self.agent_client.streaming_query_agent(api_input)
            else:
                agent_response = self.agent_client.query_agent(api_input)

            response = agent_response["response"]
            conversation_id = agent_response["conversation_id"]
            tool_calls = agent_response["tool_calls"]

            # Run all evaluations
            evaluation_results = []
            for eval_type in data_config.eval_types:
                try:
                    success = self._evaluate_single_type(
                        eval_type, data_config, response, tool_calls
                    )
                    evaluation_results.append(
                        {
                            "eval_type": eval_type,
                            "result": "PASS" if success else "FAIL",
                            "error": None,
                        }
                    )
                except (
                    ScriptExecutionError,
                    JudgeModelError,
                    ValueError,
                    AttributeError,
                    TypeError,
                ) as e:
                    logger.error(
                        "Single evaluation failed for %s (%s): %s",
                        data_config.eval_id,
                        eval_type,
                        e,
                    )
                    evaluation_results.append(
                        {"eval_type": eval_type, "result": "ERROR", "error": str(e)}
                    )

            return create_evaluation_results(
                data_config,
                response,
                evaluation_results,
                conversation_id=conversation_id,
                tool_calls=(
                    tool_calls if "tool_eval" in data_config.eval_types else None
                ),
            )

        except (AgentAPIError, ScriptExecutionError, JudgeModelError) as e:
            logger.error("Evaluation failed for %s: %s", data_config.eval_id, e)
            return create_evaluation_results(
                data_config, error_message=str(e), conversation_id=conversation_id
            )

    def _evaluate_single_type(
        self,
        eval_type: str,
        data_config: "EvaluationDataConfig",
        response: str,
        tool_calls: Optional[list[list[dict]]] = None,
    ) -> bool:
        """Evaluate single evaluation type."""
        match eval_type:
            case "action_eval":
                return self._evaluate_script(data_config)
            case "response_eval:sub-string":
                return self._evaluate_substring(data_config, response)
            case "response_eval:accuracy":
                return self._evaluate_judge_llm(data_config, response)
            # TODO(future): We should do tool_eval always ??
            case "tool_eval":
                return self._evaluate_tools(data_config, tool_calls or [])
            case _:
                logger.error("Unknown evaluation type: %s", eval_type)
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

    def _evaluate_tools(
        self,
        data_config: "EvaluationDataConfig",
        actual_tool_calls: Optional[list[list[dict]]],
    ) -> bool:
        """Evaluate using tool calls comparison."""
        if not data_config.expected_tool_calls:
            logger.error("Expected tool calls not provided for tool evaluation")
            return False

        if actual_tool_calls is None:
            logger.error("No tool calls provided for evaluation")
            return False

        return compare_tool_calls(data_config.expected_tool_calls, actual_tool_calls)

    def get_judge_manager(self) -> Optional["JudgeModelManager"]:
        """Get the judge model manager."""
        return self.judge_manager
