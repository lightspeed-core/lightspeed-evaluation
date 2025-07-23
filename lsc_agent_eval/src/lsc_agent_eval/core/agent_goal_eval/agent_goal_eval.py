"""Core orchestrator for agent goal evaluation."""

import argparse
import logging

from tqdm import tqdm

from ..utils.api_client import AgentHttpClient
from ..utils.judge import JudgeModelManager
from .eval_data import AgentGoalEvalDataManager
from .evaluator import EvaluationRunner
from .models import EvaluationDataConfig, EvaluationResult
from .results import ResultsManager

logger = logging.getLogger(__name__)


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
            self.agent_client,
            self.judge_manager,
            kubeconfig=getattr(self.eval_args, "kubeconfig", None),
        )

        # Results manager
        self.results_manager = ResultsManager(self.eval_args.result_dir)

    def run_evaluation(self) -> None:
        """Run all evaluations and save results."""
        try:
            eval_data = self.data_manager.get_eval_data()
            logger.info("Running %d evaluations", len(eval_data))

            results = []
            pbar = tqdm(eval_data)
            for data_config in pbar:
                pbar.set_description(f"Running evaluation for {data_config.eval_id}")
                result = self.evaluation_runner.run_evaluation(
                    data_config,
                    self.eval_args.agent_provider,
                    self.eval_args.agent_model,
                )
                self._print_individual_result(data_config, result, pbar)
                results.append(result)

            # Save results
            self.results_manager.save_results(results)

            # Print summary
            self._print_summary(results)

        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            raise
        finally:
            # Clean up resources
            self._cleanup()

    @staticmethod
    def _print_individual_result(
        data_config: EvaluationDataConfig, result: EvaluationResult, pbar: tqdm
    ) -> None:
        """Print individual result."""
        match result.result:
            case "PASS":
                marker = "✅"
            case "FAIL":
                marker = "❌"
            case _:
                marker = "⚠️ "
        pbar.write(f"{marker} {result.eval_id}: {result.result}")

        if result.result != "PASS":
            pbar.write(f"   Query: {result.query}")
            pbar.write(f"   Response: {result.response}")
            pbar.write(f"   Evaluation type: {result.eval_type}")
            if data_config.expected_keywords:
                pbar.write(
                    f"   Expected keywords: {','.join(data_config.expected_keywords)}"
                )
            if data_config.expected_response:
                pbar.write(f"   Expected response: {data_config.expected_response}")
            if data_config.eval_verify_script:
                pbar.write(f"   Verify script: {data_config.eval_verify_script}")
        if result.result == "ERROR":
            pbar.write(f"   Error message: {result.error}")

    def _print_summary(self, results: list[EvaluationResult]) -> None:
        """Print evaluation summary."""
        total = len(results)
        passed = sum(1 for r in results if r.result == "PASS")
        failed = sum(1 for r in results if r.result == "FAIL")
        errored = sum(1 for r in results if r.result == "ERROR")
        success_rate = (passed / total * 100) if total > 0 else 0

        print(f"\n{'='*25}")
        print("EVALUATION SUMMARY")
        print(f"{'='*25}")
        print(f"Total Evaluations: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⚠️  Errored: {errored}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"{'='*25}\n")

        self.result_summary = {"PASS": passed, "FAIL": failed, "ERROR": errored}

    def _cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self, "agent_client"):
                self.agent_client.close()
        except (AttributeError, OSError) as e:
            logger.warning("Error during cleanup: %s", e)

    def get_result_summary(self) -> dict[str, int]:
        """Get result summary."""
        return self.result_summary
