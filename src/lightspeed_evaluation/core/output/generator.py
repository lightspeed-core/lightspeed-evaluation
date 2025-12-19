"""Output and report handling - generates final results and reports."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from lightspeed_evaluation.core.constants import (
    DEFAULT_OUTPUT_DIR,
    SUPPORTED_CSV_COLUMNS,
    SUPPORTED_GRAPH_TYPES,
    SUPPORTED_OUTPUT_TYPES,
)
from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.output.statistics import (
    calculate_api_token_usage,
    calculate_basic_stats,
    calculate_detailed_stats,
)
from lightspeed_evaluation.core.output.visualization import GraphGenerator


class OutputHandler:
    """Handles output and report generation."""

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        base_filename: str = "evaluation",
        system_config: Optional[Any] = None,
    ) -> None:
        """Initialize Output handler."""
        self.output_dir = Path(output_dir)
        self.base_filename = base_filename
        self.system_config = system_config
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"✅ Output handler initialized: {self.output_dir}")

    def generate_reports(
        self,
        results: list[EvaluationResult],
        evaluation_data: Optional[list[EvaluationData]] = None,
    ) -> None:
        """Generate all output reports based on configuration.

        Args:
            results: List of evaluation results.
            evaluation_data: Optional evaluation data for API token calculation.
        """
        # Prepare timestamped base filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.base_filename}_{timestamp}"

        # Get enabled outputs from system config
        enabled_outputs = (
            self.system_config.output.enabled_outputs
            if self.system_config is not None
            else SUPPORTED_OUTPUT_TYPES
        )

        print(f"\n📊 Generating reports: {base_filename}")

        # Pre-calculate stats once for reuse across all reports
        stats = self._calculate_stats(results)

        # Calculate API token usage if evaluation data is provided
        if evaluation_data:
            stats["api_tokens"] = calculate_api_token_usage(evaluation_data)
        else:
            stats["api_tokens"] = {
                "total_api_input_tokens": 0,
                "total_api_output_tokens": 0,
                "total_api_tokens": 0,
            }

        # Generate individual reports based on configuration
        self._generate_individual_reports(
            results, base_filename, enabled_outputs, stats
        )

        # Generate graphs if enabled
        if results and (
            self.system_config is not None
            and self.system_config.visualization.enabled_graphs
        ):
            self._create_graphs(results, base_filename, stats["detailed"])

    def _calculate_stats(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Pre-calculate statistics for reuse across reports."""
        return {
            "basic": calculate_basic_stats(results),
            "detailed": (
                calculate_detailed_stats(results)
                if results
                else {"by_metric": {}, "by_conversation": {}}
            ),
        }

    def _generate_individual_reports(
        self,
        results: list[EvaluationResult],
        base_filename: str,
        enabled_outputs: list[str],
        stats: dict[str, Any],
    ) -> None:
        """Generate reports based on enabled outputs."""
        if "csv" in enabled_outputs:
            csv_file = self._generate_csv_report(results, base_filename)
            print(f"  ✅ CSV: {csv_file}")

        if "json" in enabled_outputs:
            json_file = self._generate_json_summary(
                results,
                base_filename,
                stats["basic"],
                stats["detailed"],
                stats.get("api_tokens", {}),
            )
            print(f"  ✅ JSON: {json_file}")

        if "txt" in enabled_outputs:
            txt_file = self._generate_text_summary(
                results,
                base_filename,
                stats["basic"],
                stats["detailed"],
                stats.get("api_tokens", {}),
            )
            print(f"  ✅ TXT: {txt_file}")

    def _create_graphs(
        self,
        results: list[EvaluationResult],
        base_filename: str,
        detailed_stats: dict[str, Any],
    ) -> None:
        """Create visualization graphs."""
        try:
            # Use system config visualization values or defaults
            if self.system_config is not None:
                figsize = self.system_config.visualization.figsize
                dpi = self.system_config.visualization.dpi
                enabled_graphs = self.system_config.visualization.enabled_graphs
            else:
                figsize = [12, 8]
                dpi = 300
                enabled_graphs = SUPPORTED_GRAPH_TYPES

            graph_generator = GraphGenerator(
                output_dir=str(self.output_dir), figsize=figsize, dpi=dpi
            )
            graph_files = graph_generator.generate_all_graphs(
                results, base_filename, detailed_stats, enabled_graphs
            )
            print(f"  ✅ Graphs: {len(graph_files)} files")
        except (ValueError, RuntimeError, OSError) as e:
            print(f"  ⚠️ Graph generation failed: {e}")

    def _generate_csv_report(
        self, results: list[EvaluationResult], base_filename: str
    ) -> Path:
        """Generate detailed CSV report."""
        # Move to dataframe for better aggregation
        csv_file = self.output_dir / f"{base_filename}_detailed.csv"

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Get CSV columns from system config output configuration
            csv_columns = (
                self.system_config.output.csv_columns
                if self.system_config is not None
                else SUPPORTED_CSV_COLUMNS
            )

            # Header
            writer.writerow(csv_columns)

            # Data rows
            for result in results:
                # Build row data dynamically based on configured columns
                row_data = []
                for column in csv_columns:
                    if hasattr(result, column):
                        value = getattr(result, column)
                        # Special formatting for execution_time
                        if column == "execution_time" and value is not None:
                            row_data.append(f"{value:.3f}")
                        else:
                            row_data.append(value)
                    else:
                        row_data.append("")  # Empty value for missing columns
                writer.writerow(row_data)

        return csv_file

    def _generate_json_summary(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        results: list[EvaluationResult],
        base_filename: str,
        basic_stats: dict[str, Any],
        detailed_stats: dict[str, Any],
        api_tokens: dict[str, Any],
    ) -> Path:
        """Generate JSON summary report."""
        json_file = self.output_dir / f"{base_filename}_summary.json"

        # Merge API tokens into overall stats
        judge_llm_tokens = basic_stats.get("total_judge_llm_tokens", 0)
        api_total_tokens = api_tokens.get("total_api_tokens", 0)
        overall_stats = {
            **basic_stats,
            "total_api_input_tokens": api_tokens.get("total_api_input_tokens", 0),
            "total_api_output_tokens": api_tokens.get("total_api_output_tokens", 0),
            "total_api_tokens": api_total_tokens,
            "total_tokens": judge_llm_tokens + api_total_tokens,
        }

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(results),
            "summary_stats": {
                "overall": overall_stats,
                "by_metric": detailed_stats["by_metric"],
                "by_conversation": detailed_stats["by_conversation"],
            },
            "results": [
                {
                    "conversation_group_id": r.conversation_group_id,
                    "turn_id": r.turn_id,
                    "metric_identifier": r.metric_identifier,
                    "result": r.result,
                    "score": r.score,
                    "threshold": r.threshold,
                    "execution_time": round(r.execution_time, 3),
                    "judge_llm_input_tokens": r.judge_llm_input_tokens,
                    "judge_llm_output_tokens": r.judge_llm_output_tokens,
                }
                for r in results
            ],
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return json_file

    def _generate_text_summary(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        results: list[EvaluationResult],
        base_filename: str,
        basic_stats: dict[str, Any],
        detailed_stats: dict[str, Any],
        api_tokens: dict[str, Any],
    ) -> Path:
        """Generate human-readable text summary."""
        txt_file = self.output_dir / f"{base_filename}_summary.txt"

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("LSC Evaluation Framework - Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Evaluations: {len(results)}\n\n")

            # Overall statistics
            self._write_overall_stats(f, basic_stats)

            # Token usage statistics
            self._write_token_stats(f, basic_stats, api_tokens)

            # By metric breakdown
            self._write_metric_breakdown(f, detailed_stats["by_metric"])

            # By conversation breakdown
            self._write_conversation_breakdown(f, detailed_stats["by_conversation"])

        return txt_file

    def _write_overall_stats(self, f: Any, stats: dict[str, Any]) -> None:
        """Write overall statistics section."""
        f.write("Overall Statistics:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Pass: {stats['PASS']} ({stats['pass_rate']:.1f}%)\n")
        f.write(f"Fail: {stats['FAIL']} ({stats['fail_rate']:.1f}%)\n")
        f.write(f"Error: {stats['ERROR']} ({stats['error_rate']:.1f}%)\n")
        f.write(f"Skipped: {stats['SKIPPED']} ({stats['skipped_rate']:.1f}%)\n\n")

    def _write_token_stats(
        self, f: Any, basic_stats: dict[str, Any], api_tokens: dict[str, Any]
    ) -> None:
        """Write token usage statistics section."""
        f.write("Token Usage (Judge LLM):\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input Tokens: {basic_stats['total_judge_llm_input_tokens']:,}\n")
        f.write(f"Output Tokens: {basic_stats['total_judge_llm_output_tokens']:,}\n")
        f.write(f"Total Tokens: {basic_stats['total_judge_llm_tokens']:,}\n\n")

        f.write("Token Usage (API Calls):\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input Tokens: {api_tokens.get('total_api_input_tokens', 0):,}\n")
        f.write(f"Output Tokens: {api_tokens.get('total_api_output_tokens', 0):,}\n")
        f.write(f"Total Tokens: {api_tokens.get('total_api_tokens', 0):,}\n\n")

    def _write_metric_breakdown(
        self, f: Any, by_metric: dict[str, dict[str, Any]]
    ) -> None:
        """Write metric breakdown section."""
        if not by_metric:
            return
        f.write("By Metric:\n")
        f.write("-" * 10 + "\n")
        for metric, stats in by_metric.items():
            f.write(f"{metric}:\n")
            f.write(f"  Pass: {stats['pass']} ({stats['pass_rate']:.1f}%)\n")
            f.write(f"  Fail: {stats['fail']} ({stats['fail_rate']:.1f}%)\n")
            f.write(f"  Error: {stats['error']} ({stats['error_rate']:.1f}%)\n")
            if stats.get("skipped", 0) > 0:
                f.write(
                    f"  Skipped: {stats['skipped']} ({stats['skipped_rate']:.1f}%)\n"
                )
            if "score_statistics" in stats and stats["score_statistics"]["count"] > 0:
                self._write_score_stats(f, stats["score_statistics"])
            f.write("\n")

    def _write_score_stats(self, f: Any, score_stats: dict[str, Any]) -> None:
        """Write score statistics for a metric."""
        f.write("  Score Statistics:\n")
        f.write(f"    Mean: {score_stats['mean']:.3f}\n")
        f.write(f"    Median: {score_stats['median']:.3f}\n")
        f.write(f"    Min: {score_stats['min']:.3f}, Max: {score_stats['max']:.3f}\n")
        if score_stats["count"] > 1:
            f.write(f"    Std Dev: {score_stats['std']:.3f}\n")

    def _write_conversation_breakdown(
        self, f: Any, by_conversation: dict[str, dict[str, Any]]
    ) -> None:
        """Write conversation breakdown section."""
        if not by_conversation:
            return
        f.write("By Conversation:\n")
        f.write("-" * 15 + "\n")
        for conv_id, stats in by_conversation.items():
            f.write(f"{conv_id}:\n")
            f.write(f"  Pass: {stats['pass']} ({stats['pass_rate']:.1f}%)\n")
            f.write(f"  Fail: {stats['fail']} ({stats['fail_rate']:.1f}%)\n")
            f.write(f"  Error: {stats['error']} ({stats['error_rate']:.1f}%)\n")
            if stats.get("skipped", 0) > 0:
                f.write(
                    f"  Skipped: {stats['skipped']} ({stats['skipped_rate']:.1f}%)\n"
                )
            f.write("\n")

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dir
