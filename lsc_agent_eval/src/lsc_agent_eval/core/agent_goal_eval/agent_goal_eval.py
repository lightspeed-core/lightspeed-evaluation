"""Core orchestrator for agent goal evaluation."""

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from tqdm import tqdm

from ..utils.api_client import AgentHttpClient
from ..utils.exceptions import AgentEvaluationError, ScriptExecutionError
from ..utils.judge import JudgeModelManager
from .eval_data import AgentGoalEvalDataManager
from .evaluator import EvaluationRunner
from .results import ResultsManager
from .script_runner import ScriptRunner
from .utils import create_evaluation_results

if TYPE_CHECKING:
    from .models import ConversationDataConfig, EvaluationDataConfig, EvaluationResult

logger = logging.getLogger(__name__)


def _get_result_marker(status: str) -> str:
    """Get result marker/emoji for printing."""
    match status:
        case "PASS":
            marker = "✅"
        case "FAIL":
            marker = "❌"
        case _:
            marker = "⚠️ "

    return marker


class AgentGoalEval:
    """Orchestrator for agent goal evaluation."""

    def __init__(self, eval_args: argparse.Namespace) -> None:
        """Initialize agent goal evaluation."""
        self.eval_args = eval_args
        self.result_summary: dict[str, int] = {}
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize all evaluation components."""
        # Eval data manager
        self.data_manager = AgentGoalEvalDataManager(self.eval_args.eval_data_yaml)

        # Script runner
        self.script_runner = ScriptRunner(getattr(self.eval_args, "kubeconfig", None))

        # Agent HTTP client
        self.agent_client = AgentHttpClient(
            self.eval_args.agent_endpoint, self.eval_args.agent_auth_token_file
        )

        # Judge model manager (optional)
        self.judge_manager = None
        if self.eval_args.judge_provider and self.eval_args.judge_model:
            self.judge_manager = JudgeModelManager(
                self.eval_args.judge_provider, self.eval_args.judge_model
            )

        # Evaluation runner
        self.evaluation_runner = EvaluationRunner(
            self.agent_client, self.script_runner, self.judge_manager
        )

    def run_evaluation(self) -> None:
        """Run all evaluations and save results."""
        try:
            conversations = self.data_manager.get_conversations()

            logger.info(
                "Starting Agent Goal Evaluation\n"
                "Total: %d evaluations across %d conversations",
                self.data_manager.get_eval_count(),
                len(conversations),
            )

            results = []

            # Process each conversation for evaluation
            for conv_idx, conversation in enumerate(conversations, 1):
                print(
                    f"\n📋 Conversation {conv_idx}/{len(conversations)}: "
                    f"{conversation.conversation_group}"
                )
                conversation_results = self._process_conversation(conversation)
                results.extend(conversation_results)

            # Save results
            results_manager = ResultsManager(results)
            results_manager.save_results(self.eval_args.result_dir)

            # Print summary
            self._print_summary(results_manager)

        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            raise
        finally:
            # Clean up resources
            self._cleanup()

    def _process_conversation(
        self, conversation: "ConversationDataConfig"
    ) -> list["EvaluationResult"]:
        """Process single conversation group."""
        conversation_group = conversation.conversation_group
        evaluations = conversation.conversation
        print(f"   Evaluations count: {len(evaluations)}")

        # Always start with None - conversation_id will be obtained from first API call
        conversation_id = None

        results = []

        # Run setup script for the conversation
        if conversation.setup_script:
            try:
                self._run_setup_script(conversation.setup_script, conversation_group)
            except ScriptExecutionError as e:
                # If setup fails, mark all evaluations as ERROR
                for eval_data in evaluations:
                    error_result = create_evaluation_results(
                        eval_data,
                        error_message=f"Setup script failed: {str(e)}",
                        conversation_id=conversation_id,
                    )
                    results.extend(error_result)
                print(f"❌ Setup script failed for {conversation_group}: {e}")
                return results

        # Run evaluations
        print(f"   Running {len(evaluations)} evaluations...")
        evaluation_results = self._run_conversation_evaluations(
            evaluations, conversation_group, conversation_id
        )
        results.extend(evaluation_results)

        # Run cleanup script for the conversation
        if conversation.cleanup_script:
            self._run_cleanup_script(conversation.cleanup_script, conversation_group)

        return results

    def _run_setup_script(self, setup_script: Path, conversation_group: str) -> None:
        """Run setup script for a conversation."""
        setup_success = self.script_runner.run_script(setup_script)
        if not setup_success:
            raise ScriptExecutionError("Setup script returned non-zero exit code")
        logger.debug("Setup script executed successfully for %s", conversation_group)

    def _run_cleanup_script(
        self, cleanup_script: Path, conversation_group: str
    ) -> None:
        """Run cleanup script for a conversation."""
        try:
            cleanup_success = self.script_runner.run_script(cleanup_script)
            if cleanup_success:
                logger.debug("Cleanup completed successfully")
            else:
                logger.warning("Cleanup script failed (non-critical)")
        except ScriptExecutionError as e:
            logger.warning("Cleanup script failed for %s: %s", conversation_group, e)

    def _run_conversation_evaluations(
        self,
        evaluations: list["EvaluationDataConfig"],
        conversation_group: str,
        conversation_id: Optional[str],
    ) -> list["EvaluationResult"]:
        """Run all evaluations for a conversation."""
        results = []

        with tqdm(
            total=len(evaluations),
            desc=f"Evaluating {conversation_group}",
        ) as pbar:
            for eval_data in evaluations:
                eval_results = self.evaluation_runner.run_evaluation(
                    eval_data,
                    self.eval_args.agent_provider,
                    self.eval_args.agent_model,
                    conversation_id,
                    getattr(self.eval_args, "endpoint_type", "streaming"),
                )

                # Update conversation_id from API response for subsequent evaluations
                if conversation_id is None and eval_results:
                    conversation_id = eval_results[0].conversation_id
                    print(f"  Received conversation ID from API: {conversation_id}")

                self._print_individual_results(eval_data, eval_results, pbar)
                results.extend(eval_results)

                pbar.update(1)

        return results

    @staticmethod
    def _print_individual_results(
        data_config: "EvaluationDataConfig",
        results: list["EvaluationResult"],
        pbar: tqdm,
    ) -> None:
        """Print individual results for multiple eval types."""
        if not results:
            return

        # Overall status - PASS if all pass, FAIL if any fail, ERROR if any error
        overall_result = "PASS"
        for result in results:
            if result.result == "ERROR":
                overall_result = "ERROR"
                break
            if result.result == "FAIL":
                overall_result = "FAIL"

        # Print summary line
        marker = _get_result_marker(overall_result)
        eval_types_str = ", ".join(r.eval_type for r in results)
        pbar.write(
            f"{marker} {results[0].conversation_group}/{results[0].eval_id} "
            f"[{eval_types_str}]: {overall_result}"
        )

        # Print details if not all passed
        if overall_result != "PASS":
            pbar.write(f"   Query: {results[0].query}")
            pbar.write(f"   Response: {results[0].response}")
            if any(r.tool_calls for r in results):
                tc = next((r.tool_calls for r in results if r.tool_calls), None)
                pbar.write(f"   Tool Calls: {tc}")

            # Print expected values for debugging
            if data_config.expected_keywords:
                pbar.write(
                    f"   Expected keywords: {','.join(data_config.expected_keywords)}"
                )
            if data_config.expected_response:
                pbar.write(f"   Expected response: {data_config.expected_response}")
            # Check for expected_intent in results (for intent evaluation)
            intent_result = next((r for r in results if r.expected_intent), None)
            if intent_result and intent_result.expected_intent:
                pbar.write(f"   Expected intent: {intent_result.expected_intent}")
            if data_config.expected_tool_calls:
                pbar.write(f"   Expected tool calls: {data_config.expected_tool_calls}")
            if data_config.eval_verify_script:
                pbar.write(f"   Verify script: {data_config.eval_verify_script}")

            pbar.write("   Individual results:")
            for result in results:
                result_marker = _get_result_marker(result.result)
                pbar.write(f"     {result_marker} {result.eval_type}: {result.result}")
                if result.error:
                    pbar.write(f"     Error message: {result.error}")

    def _print_summary(self, results_manager: ResultsManager) -> None:
        """Print evaluation summary."""
        stats = results_manager.get_results_stats()

        print(f"\n{'='*25}")
        print("EVALUATION SUMMARY")
        print(f"{'='*25}")
        print(f"Total Evaluations: {stats.total_evaluations}")
        print(f"✅ Passed: {stats.passed}")
        print(f"❌ Failed: {stats.failed}")
        print(f"⚠️  Errored: {stats.errored}")
        print(f"Success Rate: {stats.success_rate:.1f}%")

        # Show conversation breakdown if multiple conversations
        if len(stats.by_conversation) > 1:
            print("\nSummary by Conversation:")
            for conv_group, counts in stats.by_conversation.items():
                print(
                    f"{conv_group}: {counts['passed']}/{counts['total']} "
                    f"({counts['success_rate']:.1f}%)"
                )

        print(f"{'='*25}\n")

        self.result_summary = {
            "TOTAL": stats.total_evaluations,
            "PASS": stats.passed,
            "FAIL": stats.failed,
            "ERROR": stats.errored,
        }

    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "agent_client"):
                self.agent_client.close()
        except (AttributeError, OSError) as e:
            logger.warning("Error during cleanup: %s", e)

    def get_result_summary(self) -> dict[str, int]:
        """Get result summary."""
        if not self.result_summary:
            raise AgentEvaluationError("No results available. Run evaluation first.")

        return self.result_summary
