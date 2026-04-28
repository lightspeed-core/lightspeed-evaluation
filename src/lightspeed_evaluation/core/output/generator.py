"""Output and report handling - generates final results and reports."""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from lightspeed_evaluation.core.constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STORED_CONFIGS,
    SUPPORTED_CSV_COLUMNS,
    SUPPORTED_GRAPH_TYPES,
    SUPPORTED_OUTPUT_TYPES,
)
from lightspeed_evaluation.core.models import EvaluationData, EvaluationResult
from lightspeed_evaluation.core.models.summary import (
    ConversationStats,
    EvaluationSummary,
    MetricStats,
    OverallStats,
    SystemReport,
    StreamingStats,
    TagStats,
)
from lightspeed_evaluation.core.storage import FileBackendConfig, get_file_config
from lightspeed_evaluation.core.output.visualization import GraphGenerator

logger = logging.getLogger(__name__)


class OutputHandler:
    """Handles output and report generation."""

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        base_filename: str = "evaluation",
        system_config: Optional[Any] = None,
        file_config: Optional[FileBackendConfig] = None,
    ) -> None:
        """Initialize Output handler."""
        self.output_dir = Path(output_dir)
        self.base_filename = base_filename
        self.system_config = system_config
        self._file_config = file_config
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Output handler initialized: %s", self.output_dir)

    def _resolved_file_config(self) -> FileBackendConfig:
        """File backend settings: explicit ``file_config`` or first file entry in storage."""
        if self._file_config is not None:
            return self._file_config
        if self.system_config is not None:
            return get_file_config(self.system_config.storage)
        return FileBackendConfig()

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
        # Get quality_score_metrics from system config if available
        quality_score_metrics = None
        if (
            self.system_config is not None
            and self.system_config.quality_score is not None
        ):
            quality_score_metrics = self.system_config.quality_score.metrics

        # Build EvaluationSummary once, use it everywhere.
        # CLI path computes confidence intervals by default (when sample size > 1).
        summary = EvaluationSummary.from_results(
            results,
            evaluation_data=evaluation_data,
            compute_confidence_intervals=True,
        )

        # Generate SystemReport separately if quality score metrics are configured
        system_report = None
        if quality_score_metrics:
            system_report = SystemReport.from_results(
                summary.by_metric,
                quality_score_metrics,
            )

        # Prepare timestamped base filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.base_filename}_{timestamp}"

        # Get enabled outputs from system config
        enabled_outputs = (
            self._resolved_file_config().enabled_outputs
            if self.system_config is not None
            else SUPPORTED_OUTPUT_TYPES
        )

        logger.info("Generating reports: %s", base_filename)

        # Generate individual reports based on configuration
        self._generate_individual_reports(
            results, base_filename, enabled_outputs, summary, system_report
        )

        # Generate graphs if enabled
        if results and (
            self.system_config is not None
            and self.system_config.visualization.enabled_graphs
        ):
            self._create_graphs(results, base_filename, summary)

    def save(
        self,
        summary: EvaluationSummary,
        formats: Optional[list[str]] = None,
        output_dir: Optional[str] = None,
    ) -> list[Path]:
        """Save an EvaluationSummary to specified formats and directory.

        This is the public API for programmatic file output from a summary.

        Args:
            summary: The EvaluationSummary to persist.
            formats: List of output formats (e.g., ["csv", "json", "txt"]).
                Defaults to SUPPORTED_OUTPUT_TYPES.
            output_dir: Optional output directory override.

        Returns:
            List of generated file paths.
        """
        if formats is None:
            formats = SUPPORTED_OUTPUT_TYPES

        target_dir = Path(output_dir) if output_dir else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.base_filename}_{timestamp}"

        generated_files: list[Path] = []

        if "csv" in formats:
            csv_file = self._generate_csv_report(
                summary.results, base_filename, target_dir=target_dir
            )
            generated_files.append(csv_file)
            logger.info("CSV: %s", csv_file)

        if "json" in formats:
            json_file = self._generate_json_summary_from_model(
                summary, base_filename, target_dir=target_dir
            )
            generated_files.append(json_file)
            logger.info("JSON: %s", json_file)

        if "txt" in formats:
            txt_file = self._generate_text_summary_from_model(
                summary, base_filename, target_dir=target_dir
            )
            generated_files.append(txt_file)
            logger.info("TXT: %s", txt_file)

        return generated_files

    def _generate_individual_reports(
        self,
        results: list[EvaluationResult],
        base_filename: str,
        enabled_outputs: list[str],
        summary: EvaluationSummary,
        system_report: Optional[SystemReport] = None,
    ) -> None:
        """Generate reports based on enabled outputs."""
        if "csv" in enabled_outputs:
            csv_file = self._generate_csv_report(results, base_filename)
            logger.info("CSV: %s", csv_file)

        if "json" in enabled_outputs:
            json_file = self._generate_json_summary_from_model(summary, base_filename)
            logger.info("JSON: %s", json_file)
            # Generate system_report.json if quality score is configured
            if system_report is not None:
                system_report_file = self._generate_quality_score_report(
                    system_report, base_filename
                )
                logger.info("JSON: %s", system_report_file)

        if "txt" in enabled_outputs:
            txt_file = self._generate_text_summary_from_model(summary, base_filename)
            logger.info("TXT: %s", txt_file)

    def _create_graphs(
        self,
        results: list[EvaluationResult],
        base_filename: str,
        summary: EvaluationSummary,
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

            # Convert summary by_metric/by_conversation/by_tag to dict format
            # that the GraphGenerator expects
            detailed_stats = _summary_to_detailed_stats_dict(summary)

            graph_generator = GraphGenerator(
                output_dir=str(self.output_dir), figsize=figsize, dpi=dpi
            )
            graph_files = graph_generator.generate_all_graphs(
                results, base_filename, detailed_stats, enabled_graphs
            )
            logger.info("Graphs: %d files", len(graph_files))
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning("Graph generation failed: %s", e)

    def _generate_csv_report(
        self,
        results: list[EvaluationResult],
        base_filename: str,
        target_dir: Optional[Path] = None,
    ) -> Path:
        """Generate detailed CSV report."""
        out = target_dir if target_dir is not None else self.output_dir
        csv_file = out / f"{base_filename}_detailed.csv"

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Get CSV columns from system config storage configuration
            csv_columns = (
                self._resolved_file_config().csv_columns
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
                        # Convert judge_scores to JSON string
                        elif column == "judge_scores" and value is not None:
                            row_data.append(
                                json.dumps(
                                    [js.model_dump() for js in value], default=str
                                )
                            )
                        else:
                            row_data.append(value)
                    else:
                        row_data.append("")  # Empty value for missing columns
                writer.writerow(row_data)

        return csv_file

    def _generate_json_summary_from_model(
        self,
        summary: EvaluationSummary,
        base_filename: str,
        target_dir: Optional[Path] = None,
    ) -> Path:
        """Generate JSON summary report from an EvaluationSummary model.

        Args:
            summary: The EvaluationSummary containing all computed stats.
            base_filename: Base filename for the output file.
            target_dir: Optional directory override for output file location.

        Returns:
            Path to the generated JSON file.
        """
        out = target_dir if target_dir is not None else self.output_dir
        json_file = out / f"{base_filename}_summary.json"

        summary_stats = _build_json_summary_stats(summary)

        output = {
            "timestamp": summary.timestamp,
            "total_evaluations": len(summary.results),
            "summary_stats": summary_stats,
            "configuration": self._build_config_dict(),
            "results": [_result_to_json_dict(r) for r in summary.results],
        }

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        return json_file

    def _generate_quality_score_report(
        self,
        system_report: SystemReport,
        base_filename: str,
        target_dir: Optional[Path] = None,
    ) -> Path:
        """Generate quality score JSON report.

        Args:
            system_report: The SystemReport model instance.
            base_filename: Base filename for the output file.
            target_dir: Optional directory override for output file location.

        Returns:
            Path to the generated quality_score.json file.
        """
        out = target_dir if target_dir is not None else self.output_dir
        quality_score_file = out / f"{base_filename}_system_report.json"

        output = {
            "timestamp": datetime.now().isoformat(),
            "aggregated_quality_score": system_report.aggregated_quality_score,
            "quality_metrics": {
                metric_id: {
                    "mean": stats.mean,
                    "count": stats.count,
                    "weight": stats.weight,
                }
                for metric_id, stats in system_report.quality_metrics.items()
            },
            "extra_metrics": {
                metric_id: {
                    "mean": stats.mean,
                    "count": stats.count,
                }
                for metric_id, stats in system_report.extra_metrics.items()
            },
            "api_latency": system_report.api_latency,
            "api_tokens": system_report.api_tokens,
        }

        with open(quality_score_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)

        return quality_score_file

    def _generate_text_summary_from_model(
        self,
        summary: EvaluationSummary,
        base_filename: str,
        target_dir: Optional[Path] = None,
    ) -> Path:
        """Generate human-readable text summary from an EvaluationSummary model.

        Args:
            summary: The EvaluationSummary containing all computed stats.
            base_filename: Base filename for the output file.
            target_dir: Optional directory override for output file location.

        Returns:
            Path to the generated text file.
        """
        out = target_dir if target_dir is not None else self.output_dir
        txt_file = out / f"{base_filename}_summary.txt"

        # Build compatible dicts from summary model
        basic_stats = _overall_to_basic_stats_dict(summary.overall)
        api_tokens = (
            summary.api_tokens.model_dump()
            if summary.api_tokens
            else {
                "total_api_input_tokens": 0,
                "total_api_output_tokens": 0,
                "total_api_tokens": 0,
            }
        )
        streaming_stats = (
            _streaming_stats_to_dict(summary.streaming) if summary.streaming else {}
        )
        detailed_stats = _summary_to_detailed_stats_dict(summary)

        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("LSC Evaluation Framework - Summary Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Evaluations: {len(summary.results)}\n\n")

            # Overall statistics
            self._write_overall_stats(f, basic_stats)

            # Token usage statistics
            self._write_token_stats(f, basic_stats, api_tokens)

            # Streaming performance statistics
            self._write_streaming_stats(f, streaming_stats)

            # Breakdowns by category
            self._write_breakdown_section(
                f, "By Metric", detailed_stats["by_metric"], include_scores=True
            )
            self._write_breakdown_section(
                f, "By Conversation", detailed_stats["by_conversation"]
            )
            self._write_breakdown_section(
                f, "By Tag", detailed_stats.get("by_tag", {}), include_scores=True
            )

            # Configuration parameters
            self._write_config_params(f)

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

        f.write("Token Usage (Embeddings):\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input Tokens: {basic_stats['total_embedding_tokens']:,}\n\n")

        f.write("Token Usage (API Calls):\n")
        f.write("-" * 20 + "\n")
        f.write(f"Input Tokens: {api_tokens.get('total_api_input_tokens', 0):,}\n")
        f.write(f"Output Tokens: {api_tokens.get('total_api_output_tokens', 0):,}\n")
        f.write(f"Total Tokens: {api_tokens.get('total_api_tokens', 0):,}\n\n")

    def _write_streaming_stats(self, f: Any, streaming_stats: dict[str, Any]) -> None:
        """Write streaming performance statistics section."""
        # Check if there are any streaming metrics
        ttft = streaming_stats.get("time_to_first_token", {})
        duration = streaming_stats.get("streaming_duration", {})
        throughput = streaming_stats.get("tokens_per_second", {})

        if ttft.get("count", 0) == 0:
            return  # No streaming data available

        f.write("Streaming Performance:\n")
        f.write("-" * 20 + "\n")

        self._write_numeric_stats(f, "Time to First Token (seconds)", ttft)
        self._write_numeric_stats(f, "Streaming Duration (seconds)", duration)
        self._write_numeric_stats(f, "Tokens per Second", throughput, precision=1)

        f.write("\n")

    def _write_numeric_stats(
        self,
        f: Any,
        title: str,
        stats: dict[str, Any],
        *,
        precision: int = 3,
    ) -> None:
        """Write numeric statistics with mean, median, min, max.

        Args:
            f: File handle to write to.
            title: Section title.
            stats: Statistics dictionary with mean, median, min, max, count.
            precision: Decimal precision for formatting numbers.
        """
        if stats.get("count", 0) == 0:
            return

        fmt = f".{precision}f"
        f.write(f"{title}:\n")
        f.write(f"  Mean: {stats['mean']:{fmt}}\n")
        f.write(f"  Median: {stats['median']:{fmt}}\n")
        f.write(f"  Min: {stats['min']:{fmt}}, Max: {stats['max']:{fmt}}\n")

    def _write_breakdown_section(
        self,
        f: Any,
        title: str,
        stats_dict: dict[str, dict[str, Any]],
        *,
        include_scores: bool = False,
    ) -> None:
        """Write a breakdown section by different categories.

        Args:
            f: File handle to write to.
            title: Section title (e.g., "By Metric", "By Conversation").
            stats_dict: Dictionary mapping keys to their statistics.
            include_scores: Whether to include score statistics.
        """
        if not stats_dict:
            return
        f.write(f"{title}:\n")
        f.write("-" * len(title) + "\n")
        for key, stats in stats_dict.items():
            f.write(f"{key}:\n")
            f.write(f"  Pass: {stats['pass']} ({stats['pass_rate']:.1f}%)\n")
            f.write(f"  Fail: {stats['fail']} ({stats['fail_rate']:.1f}%)\n")
            f.write(f"  Error: {stats['error']} ({stats['error_rate']:.1f}%)\n")
            if stats.get("skipped", 0) > 0:
                f.write(
                    f"  Skipped: {stats['skipped']} ({stats['skipped_rate']:.1f}%)\n"
                )
            if (
                include_scores
                and "score_statistics" in stats
                and stats["score_statistics"]["count"] > 0
            ):
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

    def _write_config_params(self, f: Any) -> None:
        """Write configuration parameters section to file.

        Args:
            f: File handle to write to.
        """
        if self.system_config is None:
            return

        f.write("Configuration Parameters:\n")
        f.write("-" * 25 + "\n")

        # Get configured sections to include
        included_sections = self._get_included_config_sections()

        # Iterate through specified configuration sections
        for field_name in self.system_config.model_fields.keys():
            # Skip sections not in the included list
            if field_name not in included_sections:
                continue

            field_value = getattr(self.system_config, field_name)

            # Format section name nicely
            section_name = field_name.replace("_", " ").title()
            f.write(f"\n{section_name}:\n")

            # Use dynamic formatter for the section
            lines = self._format_config_section(field_value, indent=2)
            for line in lines:
                f.write(f"{line}\n")

        f.write("\n")

    def _get_included_config_sections(self) -> list[str]:
        """Get list of configuration sections to include in summaries.

        Returns:
            List of section names (e.g., ['llm', 'embedding', 'api']).
        """
        if self.system_config is not None and hasattr(self.system_config, "storage"):
            return self._resolved_file_config().summary_config_sections
        return DEFAULT_STORED_CONFIGS

    def _convert_config_to_dict(self, config: BaseModel | dict) -> dict[str, Any]:
        """Convert configuration to dictionary, excluding sensitive fields.

        Args:
            config: Configuration object (Pydantic model or dict).

        Returns:
            Dictionary with cache_dir excluded.
        """
        if isinstance(config, BaseModel):
            return config.model_dump(exclude={"cache_dir"})
        return {k: v for k, v in config.items() if k != "cache_dir"}

    def _format_config_section(
        self, config: BaseModel | dict, indent: int = 2
    ) -> list[str]:
        """Format configuration section for text output.

        Args:
            config: Configuration object (Pydantic model or dict).
            indent: Number of spaces for indentation.

        Returns:
            List of formatted strings.
        """
        lines = []
        prefix = " " * indent

        # Convert Pydantic model to dict
        config_dict = self._convert_config_to_dict(config)

        # Format each field
        for key, value in config_dict.items():
            display_name = key.replace("_", " ").title()

            # Handle lists (like enabled_outputs)
            if isinstance(value, list):
                if value:  # Only show non-empty lists
                    lines.append(
                        f"{prefix}{display_name}: {', '.join(map(str, value))}"
                    )
            # Handle None values
            elif value is None:
                lines.append(f"{prefix}{display_name}: None")
            # Handle everything else (str, int, float, bool)
            else:
                lines.append(f"{prefix}{display_name}: {value}")

        return lines

    def _build_config_dict(self) -> dict[str, Any]:
        """Build configuration parameters dictionary for JSON output.

        Returns:
            Dictionary mapping section names to their configuration values.
        """
        if self.system_config is None:
            return {}

        config_dict = {}
        included_sections = self._get_included_config_sections()

        for field_name in self.system_config.model_fields.keys():
            if field_name not in included_sections:
                continue

            field_value = getattr(self.system_config, field_name)

            # Convert Pydantic model to dict, excluding cache_dir for security
            config_dict[field_name] = self._convert_config_to_dict(field_value)

        return config_dict

    def get_output_directory(self) -> Path:
        """Get the output directory path."""
        return self.output_dir


def _build_json_summary_stats(summary: EvaluationSummary) -> dict[str, Any]:
    """Build the summary_stats dict for JSON output from an EvaluationSummary.

    Merges overall stats with API token usage and builds per-metric,
    per-conversation, per-tag, and streaming performance sections.

    Args:
        summary: The EvaluationSummary instance.

    Returns:
        Dictionary matching the JSON summary format.
    """
    overall = summary.overall
    api_tokens = summary.api_tokens
    judge_tokens = overall.total_judge_llm_tokens
    api_total = api_tokens.total_api_tokens if api_tokens else 0

    overall_stats = {
        **_overall_to_basic_stats_dict(overall),
        "total_api_input_tokens": (
            api_tokens.total_api_input_tokens if api_tokens else 0
        ),
        "total_api_output_tokens": (
            api_tokens.total_api_output_tokens if api_tokens else 0
        ),
        "total_api_tokens": api_total,
        "total_tokens": judge_tokens + api_total,
    }

    result: dict[str, Any] = {
        "overall": overall_stats,
        "by_metric": _metric_stats_to_dict(summary.by_metric),
        "by_conversation": _conversation_stats_to_dict(summary.by_conversation),
        "by_tag": _tag_stats_to_dict(summary.by_tag),
    }

    if summary.streaming is not None:
        result["streaming_performance"] = _streaming_stats_to_dict(summary.streaming)

    return result


def _result_to_json_dict(r: EvaluationResult) -> dict[str, Any]:
    """Convert a single EvaluationResult to JSON-serializable dict.

    Args:
        r: The evaluation result to convert.

    Returns:
        Dictionary matching the existing JSON summary result format.
    """
    return {
        "conversation_group_id": r.conversation_group_id,
        "tag": r.tag,
        "turn_id": r.turn_id,
        "metric_identifier": r.metric_identifier,
        "result": r.result,
        "score": r.score,
        "threshold": r.threshold,
        "execution_time": round(r.execution_time, 3),
        "judge_llm_input_tokens": r.judge_llm_input_tokens,
        "judge_llm_output_tokens": r.judge_llm_output_tokens,
        "judge_scores": (
            [js.model_dump() for js in r.judge_scores] if r.judge_scores else None
        ),
        "time_to_first_token": r.time_to_first_token,
        "streaming_duration": r.streaming_duration,
        "tokens_per_second": r.tokens_per_second,
    }


def _overall_to_basic_stats_dict(
    overall: "OverallStats",
) -> dict[str, Any]:
    """Convert OverallStats to the dict format expected by text output.

    Args:
        overall: OverallStats model instance.

    Returns:
        Dictionary with keys matching the original calculate_basic_stats format.
    """
    return {
        "TOTAL": overall.total,
        "PASS": overall.passed,
        "FAIL": overall.failed,
        "ERROR": overall.error,
        "SKIPPED": overall.skipped,
        "pass_rate": overall.pass_rate,
        "fail_rate": overall.fail_rate,
        "error_rate": overall.error_rate,
        "skipped_rate": overall.skipped_rate,
        "total_judge_llm_input_tokens": overall.total_judge_llm_input_tokens,
        "total_judge_llm_output_tokens": overall.total_judge_llm_output_tokens,
        "total_judge_llm_tokens": overall.total_judge_llm_tokens,
        "total_embedding_tokens": overall.total_embedding_tokens,
    }


def _group_stats_to_dict(
    stats: MetricStats | ConversationStats | TagStats,
) -> dict[str, Any]:
    """Convert a group stats model to the dict format for text output.

    Args:
        stats: MetricStats, ConversationStats, or TagStats instance.

    Returns:
        Dictionary with lowercase keys matching original detailed stats format.
    """
    result: dict[str, Any] = {
        "pass": stats.passed,
        "fail": stats.failed,
        "error": stats.error,
        "skipped": stats.skipped,
        "pass_rate": stats.pass_rate,
        "fail_rate": stats.fail_rate,
        "error_rate": stats.error_rate,
        "skipped_rate": stats.skipped_rate,
    }
    if (
        isinstance(stats, (MetricStats, TagStats))
        and stats.score_statistics is not None
    ):
        score_stats = stats.score_statistics
        result["score_statistics"] = {
            "count": score_stats.count,
            "mean": score_stats.mean,
            "median": score_stats.median,
            "std": score_stats.std,
            "min": score_stats.min_score,
            "max": score_stats.max_score,
            "confidence_interval": score_stats.confidence_interval,
        }
    return result


def _metric_stats_to_dict(
    by_metric: dict[str, MetricStats],
) -> dict[str, dict[str, Any]]:
    """Convert by_metric model dict to legacy dict format.

    Args:
        by_metric: Dictionary mapping metric IDs to MetricStats models.

    Returns:
        Dictionary in the original detailed stats format.
    """
    return {k: _group_stats_to_dict(v) for k, v in by_metric.items()}


def _conversation_stats_to_dict(
    by_conversation: dict[str, ConversationStats],
) -> dict[str, dict[str, Any]]:
    """Convert by_conversation model dict to legacy dict format.

    Args:
        by_conversation: Dictionary mapping conversation IDs to ConversationStats.

    Returns:
        Dictionary in the original detailed stats format.
    """
    return {k: _group_stats_to_dict(v) for k, v in by_conversation.items()}


def _tag_stats_to_dict(
    by_tag: dict[str, TagStats],
) -> dict[str, dict[str, Any]]:
    """Convert by_tag model dict to legacy dict format.

    Args:
        by_tag: Dictionary mapping tags to TagStats models.

    Returns:
        Dictionary in the original detailed stats format.
    """
    return {k: _group_stats_to_dict(v) for k, v in by_tag.items()}


def _summary_to_detailed_stats_dict(
    summary: EvaluationSummary,
) -> dict[str, Any]:
    """Convert EvaluationSummary to the detailed stats dict format.

    This produces a dictionary with by_metric, by_conversation, by_tag keys
    matching the format from calculate_detailed_stats().

    Args:
        summary: The EvaluationSummary instance.

    Returns:
        Dictionary matching the original detailed stats format.
    """
    return {
        "by_metric": _metric_stats_to_dict(summary.by_metric),
        "by_conversation": _conversation_stats_to_dict(summary.by_conversation),
        "by_tag": _tag_stats_to_dict(summary.by_tag),
    }


def _streaming_stats_to_dict(streaming: StreamingStats) -> dict[str, Any]:
    """Convert StreamingStats model to the dict format for text output.

    Args:
        streaming: StreamingStats model instance.

    Returns:
        Dictionary matching the original streaming stats format.
    """
    result: dict[str, Any] = {}
    for field_name in (
        "time_to_first_token",
        "streaming_duration",
        "tokens_per_second",
    ):
        numeric = getattr(streaming, field_name, None)
        if numeric is not None:
            result[field_name] = {
                "count": numeric.count,
                "mean": numeric.mean,
                "median": numeric.median,
                "std": numeric.std,
                "min": numeric.min_value,
                "max": numeric.max_value,
            }
        else:
            result[field_name] = {"count": 0}
    return result
