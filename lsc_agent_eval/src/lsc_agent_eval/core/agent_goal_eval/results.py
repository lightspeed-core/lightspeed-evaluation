"""Results management for agent evaluation."""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..utils.exceptions import AgentEvaluationError
from .models import EvaluationResult, EvaluationStats

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages evaluation results and output."""

    def __init__(self, results: list[EvaluationResult]):
        """Initialize results manager."""
        self.results = results

        self.results_stats = EvaluationStats.from_results(results)

    def save_results(self, result_dir: str) -> None:
        """Save evaluation results/statistics to CSV and JSON files."""
        if not self.results:
            logger.warning("No result to save")
            return

        try:
            output_dir = Path(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = output_dir / f"agent_goal_eval_results_{timestamp}.csv"
            json_file = output_dir / f"agent_goal_eval_summary_{timestamp}.json"

            # Save CSV results
            self._save_csv_results(csv_file)
            # Save summary JSON
            self._save_json_summary(json_file)

        except Exception as e:
            logger.error("Failed to save results: %s", e)
            raise AgentEvaluationError(f"Failed to save results: {e}") from e

    def _save_csv_results(self, file_path: Path) -> None:
        """Save results to CSV file."""
        data = []
        for result in self.results:
            data.append(
                {
                    "conversation_group": result.conversation_group,
                    "conversation_id": result.conversation_id,
                    "eval_id": result.eval_id,
                    "query": result.query,
                    "response": result.response,
                    "eval_type": result.eval_type,
                    "result": result.result,
                    "error": result.error,
                }
            )

        df = pd.DataFrame(data)

        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info("Results saved to %s", file_path)

    def _save_json_summary(self, file_path: Path) -> None:
        """Save eval summary to JSON file."""
        statistics = {
            "summary": {
                "total_evaluations": self.results_stats.total_evaluations,
                "total_conversations": self.results_stats.total_conversations,
                "passed": self.results_stats.passed,
                "failed": self.results_stats.failed,
                "errored": self.results_stats.errored,
                "success_rate": round(self.results_stats.success_rate, 2),
            },
            "by_conversation": self.results_stats.by_conversation,
            "by_eval_type": self.results_stats.by_eval_type,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        logger.info("Summary saved to %s", file_path)

    def get_results_stats(self) -> EvaluationStats:
        """Get result stats/summary."""
        return self.results_stats
