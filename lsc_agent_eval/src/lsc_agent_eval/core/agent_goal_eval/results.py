"""Results management for agent evaluation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .models import EvaluationResult

logger = logging.getLogger(__name__)


class ResultsManager:
    """Manages evaluation results and output."""

    def __init__(self, result_dir: str):
        """Initialize results manager."""
        self.result_dir = result_dir
        self.result_path = Path(result_dir)

    def save_results(
        self,
        results: list[EvaluationResult],
        filename: Optional[str] = None,
    ) -> None:
        """Save evaluation results to CSV file."""
        # Create directory if it doesn't exist
        self.result_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_goal_eval_results_{timestamp}.csv"

        # Create full file path
        file_path = self.result_path / filename

        # Convert results to DataFrame
        data = []
        for result in results:
            data.append(
                {
                    "eval_id": result.eval_id,
                    "query": result.query,
                    "response": result.response,
                    "eval_type": result.eval_type,
                    "result": result.result,
                    "error": result.error or "",
                }
            )

        df = pd.DataFrame(data)

        # Save to CSV using pandas
        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info("Results saved to %s", file_path)

    def get_output_dir(self) -> str:
        """Get the output directory path."""
        return str(self.result_path)
