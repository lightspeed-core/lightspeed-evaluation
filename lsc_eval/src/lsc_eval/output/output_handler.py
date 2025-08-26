"""Output and report handling - generates final results and reports."""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..core.models import EvaluationResult
from ..core.config_loader import DEFAULT_CSV_COLUMNS
from .utils import calculate_basic_stats, calculate_detailed_stats
from .visualization import GraphGenerator


class OutputHandler:
    """Handles output and report generation."""

    def __init__(
        self,
        output_dir: str = "./eval_output",
        base_filename: str = "evaluation",
        system_config=None,
    ):
        """Initialize Output handler."""
        self.output_dir = Path(output_dir)
        self.base_filename = base_filename
        self.system_config = system_config
        self.output_dir.mkdir(exist_ok=True)

        print(f"✅ Output handler initialized: {self.output_dir}")

    def generate_reports(
        self, results: List[EvaluationResult], include_graphs: bool = True
    ) -> None:
        """Generate all output reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.base_filename}_{timestamp}"

        print(f"\n📊 Generating reports: {base_filename}")

        # Pre-calculate stats once for reuse across all reports
        basic_stats = calculate_basic_stats(results)
        detailed_stats = (
            calculate_detailed_stats(results)
            if results
            else {"by_metric": {}, "by_conversation": {}}
        )

        # Generate CSV report
        csv_file = self._generate_csv_report(results, base_filename)
        print(f"  ✅ CSV: {csv_file}")

        # Generate JSON summary (pass pre-calculated stats)
        json_file = self._generate_json_summary(results, base_filename, basic_stats, detailed_stats)
        print(f"  ✅ JSON: {json_file}")

        # Generate text summary (pass pre-calculated stats)
        txt_file = self._generate_text_summary(results, base_filename, basic_stats, detailed_stats)
        print(f"  ✅ TXT: {txt_file}")

        # Generate graphs if we have results (pass pre-calculated stats)
        if results and include_graphs:
            try:
                # Use system config values or fallback to defaults
                figsize = getattr(self.system_config, "visualization_figsize", [12, 8])
                dpi = getattr(self.system_config, "visualization_dpi", 300)

                graph_generator = GraphGenerator(
                    output_dir=str(self.output_dir), figsize=figsize, dpi=dpi
                )
                graph_files = graph_generator.generate_all_graphs(
                    results, base_filename, detailed_stats
                )
                print(f"  ✅ Graphs: {len(graph_files)} files")
            except (ValueError, RuntimeError, OSError) as e:
                print(f"  ⚠️ Graph generation failed: {e}")

    def _generate_csv_report(self, results: List[EvaluationResult], base_filename: str) -> Path:
        """Generate detailed CSV report."""
        # Move to dataframe for better aggregation
        csv_file = self.output_dir / f"{base_filename}_detailed.csv"

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Get CSV columns from system config, with fallback to defaults
            csv_columns = getattr(
                self.system_config,
                "csv_columns",
                DEFAULT_CSV_COLUMNS + ["query", "response"],
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

    def _generate_json_summary(
        self,
        results: List[EvaluationResult],
        base_filename: str,
        basic_stats: Dict[str, Any],
        detailed_stats: Dict[str, Any],
    ) -> Path:
        """Generate JSON summary report."""
        json_file = self.output_dir / f"{base_filename}_summary.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(results),
            "summary_stats": {
                "overall": basic_stats,
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
                }
                for r in results
            ],
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return json_file

    def _generate_text_summary(
        self,
        results: List[EvaluationResult],
        base_filename: str,
        basic_stats: Dict[str, Any],
        detailed_stats: Dict[str, Any],
    ) -> Path:
        """Generate human-readable text summary."""
        txt_file = self.output_dir / f"{base_filename}_summary.txt"

        stats = {
            "overall": basic_stats,
            "by_metric": detailed_stats["by_metric"],
            "by_conversation": detailed_stats["by_conversation"],
        }

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("LSC Evaluation Framework - Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Evaluations: {len(results)}\n\n")

            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Pass: {stats['overall']['PASS']} ({stats['overall']['pass_rate']:.1f}%)\n")
            f.write(f"Fail: {stats['overall']['FAIL']} ({stats['overall']['fail_rate']:.1f}%)\n")
            f.write(
                f"Error: {stats['overall']['ERROR']} ({stats['overall']['error_rate']:.1f}%)\n\n"
            )

            # By metric breakdown
            if stats["by_metric"]:
                f.write("By Metric:\n")
                f.write("-" * 10 + "\n")
                for metric, metric_stats in stats["by_metric"].items():
                    f.write(f"{metric}:\n")
                    f.write(f"  Pass: {metric_stats['pass']} ({metric_stats['pass_rate']:.1f}%)\n")
                    f.write(f"  Fail: {metric_stats['fail']} ({metric_stats['fail_rate']:.1f}%)\n")
                    f.write(
                        f"  Error: {metric_stats['error']} ({metric_stats['error_rate']:.1f}%)\n"
                    )
                    if (
                        "score_statistics" in metric_stats
                        and metric_stats["score_statistics"]["count"] > 0
                    ):
                        score_stats = metric_stats["score_statistics"]
                        f.write("  Score Statistics:\n")
                        f.write(f"    Mean: {score_stats['mean']:.3f}\n")
                        f.write(f"    Median: {score_stats['median']:.3f}\n")
                        f.write(
                            f"    Min: {score_stats['min']:.3f}, Max: {score_stats['max']:.3f}\n"
                        )
                        if score_stats["count"] > 1:
                            f.write(f"    Std Dev: {score_stats['std']:.3f}\n")
                    f.write("\n")

            # By conversation breakdown
            if stats["by_conversation"]:
                f.write("By Conversation:\n")
                f.write("-" * 15 + "\n")
                for conv_id, conv_stats in stats["by_conversation"].items():
                    f.write(f"{conv_id}:\n")
                    f.write(f"  Pass: {conv_stats['pass']} ({conv_stats['pass_rate']:.1f}%)\n")
                    f.write(f"  Fail: {conv_stats['fail']} ({conv_stats['fail_rate']:.1f}%)\n")
                    f.write(f"  Error: {conv_stats['error']} ({conv_stats['error_rate']:.1f}%)\n")
                    f.write("\n")

        return txt_file

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dir
