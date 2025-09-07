"""Visualization module for generating evaluation graphs."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BASE_COLORS

from ..constants import DEFAULT_OUTPUT_DIR
from ..models import EvaluationResult
from .statistics import calculate_basic_stats, calculate_detailed_stats

CHART_COLORS = {
    "PASS": "#28a745",  # Green
    "FAIL": "#dc3545",  # Red
    "ERROR": "#ffc107",  # Yellow/Orange
}


class GraphGenerator:
    """Handles generation of evaluation visualization graphs."""

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        figsize: Optional[List[int]] = None,
        dpi: int = 300,
    ):
        """Initialize Graph generator."""
        self.output_dir = Path(output_dir)
        self.graphs_dir = self.output_dir / "graphs"
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize if figsize else [12, 8]
        self.dpi = dpi
        self.logger = logging.getLogger("lightspeed_evaluation.GraphGenerator")

        # Set basic matplotlib settings
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.alpha"] = 0.3
        sns.set_palette("husl")

    def _calculate_summary_stats(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Calculate summary statistics from results."""
        return calculate_basic_stats(results)

    def _group_results_by_metric(
        self, results: List[EvaluationResult]
    ) -> Dict[str, List[float]]:
        """Group results by metric identifier."""
        metric_groups: Dict[str, List[float]] = {}
        for result in results:
            if result.score is not None:
                if result.metric_identifier not in metric_groups:
                    metric_groups[result.metric_identifier] = []
                metric_groups[result.metric_identifier].append(result.score)
        return metric_groups

    def generate_all_graphs(
        self,
        results: List[EvaluationResult],
        base_filename: str,
        detailed_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Generate all visualization graphs."""
        graph_files = {}

        try:
            self.logger.debug("Generating graphs for %d results", len(results))

            # Use pre-calculated stats if provided, otherwise calculate them
            summary_stats = (
                detailed_stats
                if detailed_stats is not None
                else self._calculate_detailed_summary_stats(results)
            )

            if not summary_stats["by_metric"]:
                self.logger.warning("No metric data available for graph generation")
                return {}

            # 1. Pass rates graph
            pass_rates_file = self._generate_pass_rates_graph(
                summary_stats["by_metric"], base_filename
            )
            if pass_rates_file:
                graph_files["pass_rates"] = str(pass_rates_file)

            # 2. Score distribution graph
            score_dist_file = self._generate_score_distribution_graph(
                results, base_filename
            )
            if score_dist_file:
                graph_files["score_distribution"] = str(score_dist_file)

            # 3. Status breakdown pie chart
            pie_chart_file = self._generate_status_breakdown_pie_chart(
                results, base_filename
            )
            if pie_chart_file:
                graph_files["status_breakdown"] = str(pie_chart_file)

            # 4. Conversation heatmap (only if multiple conversations)
            if len(summary_stats["by_conversation"]) > 1:
                heatmap_file = self._generate_conversation_heatmap(
                    results, base_filename
                )
                if heatmap_file:
                    graph_files["conversation_heatmap"] = str(heatmap_file)

        except (ValueError, RuntimeError, OSError) as e:
            self.logger.error("Graph generation error: %s", e, exc_info=True)

        self.logger.info("Generated %d graphs", len(graph_files))
        return graph_files

    def get_supported_graph_types(self) -> List[str]:
        """Get list of supported graph types."""
        return [
            "pass_rates",
            "score_distribution",
            "status_breakdown",
            "conversation_heatmap",
        ]

    def _generate_pass_rates_graph(
        self, by_metric_stats: Dict[str, Any], base_filename: str
    ) -> Path:
        """Generate pass rates bar chart."""
        # Prepare data
        metrics = []
        pass_rates = []
        status_breakdowns = []

        for metric, stats in by_metric_stats.items():
            metrics.append(metric)
            pass_rates.append(stats["pass_rate"])
            status_breakdowns.append(
                f"P:{stats['pass']} F:{stats['fail']} E:{stats['error']}"
            )

        # Create figure
        _, ax = plt.subplots(figsize=(12, 8))

        # Create bar chart
        bars = ax.bar(metrics, pass_rates, color="skyblue", alpha=0.7)

        # Add percentage labels with status breakdown
        for chart_bar, breakdown in zip(bars, status_breakdowns):
            height = chart_bar.get_height()
            ax.text(
                chart_bar.get_x() + chart_bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%\n({breakdown})",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Customize chart
        ax.set_title("Pass Rates by Metric", fontsize=16, fontweight="bold")
        ax.set_xlabel("Pass Rate (%)", fontsize=12)
        ax.set_ylabel("Metrics", fontsize=12)
        ax.set_ylim(0, 105)  # Give space for labels

        # Rotate x-axis labels if needed
        if len(metrics) > 5:
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        # Save
        filename = self.graphs_dir / f"{base_filename}_pass_rates.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def _generate_score_distribution_graph(  # pylint: disable=too-many-locals
        self, results: List[EvaluationResult], base_filename: str
    ) -> Optional[Path]:
        """Generate score distribution box plot with quartile backgrounds."""
        # Prepare data - only include results with valid scores
        valid_results = [r for r in results if r.score is not None]
        if not valid_results:
            self.logger.warning("No valid scores for score distribution graph")
            return None

        # Group by metric
        metric_groups: Dict[str, List[float]] = {}
        for result in valid_results:
            if result.metric_identifier not in metric_groups:
                metric_groups[result.metric_identifier] = []
            if result.score is not None:  # Additional check for pyright
                metric_groups[result.metric_identifier].append(result.score)

        if not metric_groups:
            return None

        # Convert to DataFrame with equal-length arrays (pad with NaN)
        max_len = max(len(scores) for scores in metric_groups.values())
        score_data = {}
        for metric_id, scores in metric_groups.items():
            padded_scores = scores + [np.nan] * (max_len - len(scores))
            score_data[metric_id] = padded_scores

        results_df = pd.DataFrame(score_data)

        _, ax = plt.subplots(figsize=tuple(self.figsize), dpi=self.dpi)
        ax.set_xlabel("Score", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1)

        # Add quartile background spans (like old unified_ls_eval)
        ax.axvspan(0, 0.25, facecolor="gainsboro")  # Gray for 0-25%
        ax.axvspan(0.25, 0.5, facecolor="mistyrose")  # Light red for 25-50%
        ax.axvspan(0.5, 0.75, facecolor="lightyellow")  # Light yellow for 50-75%
        ax.axvspan(0.75, 1.0, facecolor="lightgreen")  # Light green for 75-100%

        # Add vertical quartile lines
        ax.axvline(x=0.25, linewidth=2, color="red")
        ax.axvline(x=0.5, linewidth=2, color="orange")
        ax.axvline(x=0.75, linewidth=2, color="green")

        ax.grid(True)

        # Box plot
        bplot = ax.boxplot(
            results_df.fillna(0),
            sym=".",
            widths=0.5,
            vert=False,
            patch_artist=True,
        )

        labels = results_df.columns

        colors = list(BASE_COLORS.keys())[: len(labels)]
        for patch, color in zip(bplot["boxes"], colors):
            patch.set_facecolor(color)

        # Set y-tick labels
        ax.set_yticklabels(labels)

        plt.yticks(rotation=45)

        ax.set_title("Score Distribution by Metric", fontsize=16, fontweight="bold")

        plt.tight_layout()

        # Save
        filename = self.graphs_dir / f"{base_filename}_score_distribution.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return filename

    def _generate_status_breakdown_pie_chart(  # pylint: disable=too-many-locals
        self, results: List[EvaluationResult], base_filename: str
    ) -> Optional[Path]:
        """Generate pie chart showing overall pass/fail/error breakdown."""
        if not results:
            return None

        # Count status breakdown
        status_counts = {"PASS": 0, "FAIL": 0, "ERROR": 0}
        for result in results:
            if result.result in status_counts:
                status_counts[result.result] += 1

        # Filter out zero counts
        filtered_counts = {k: v for k, v in status_counts.items() if v > 0}

        if not filtered_counts:
            return None

        # Prepare data
        labels = list(filtered_counts.keys())
        sizes = list(filtered_counts.values())
        total = sum(sizes)

        pie_colors = [CHART_COLORS.get(label, "#6c757d") for label in labels]

        _, ax = plt.subplots(figsize=tuple(self.figsize), dpi=self.dpi)

        # Create pie chart
        explode_values = [0.05] * len(labels)
        pie_result = ax.pie(
            sizes,
            labels=labels,
            colors=pie_colors,
            autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*total)})",
            startangle=90,
            explode=explode_values,
        )

        # Handle autopct texts if they exist
        if len(pie_result) == 3:
            _, texts, autotexts = pie_result
            for autotext in autotexts:
                autotext.set_color("white")
                autotext.set_fontsize(10)
                autotext.set_fontweight("bold")
        else:
            _, texts = pie_result

        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight("bold")

        ax.set_title(
            f"Evaluation Results Breakdown\nTotal: {total} evaluations",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        ax.axis("equal")

        plt.tight_layout()

        # Save
        filename = self.graphs_dir / f"{base_filename}_status_breakdown.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return filename

    def _generate_conversation_heatmap(  # pylint: disable=too-many-locals
        self, results: List[EvaluationResult], base_filename: str
    ) -> Optional[Path]:
        """Generate conversation-level heatmap showing pass rates for each metric."""
        # Build conversation vs metric matrix
        # (only include existing metric-conversation combinations)
        conversation_metrics: Dict[str, Dict[str, Dict[str, int]]] = {}

        for result in results:
            conv_id = result.conversation_group_id
            metric_id = result.metric_identifier

            if conv_id not in conversation_metrics:
                conversation_metrics[conv_id] = {}

            if metric_id not in conversation_metrics[conv_id]:
                conversation_metrics[conv_id][metric_id] = {
                    "pass": 0,
                    "fail": 0,
                    "error": 0,
                }

            if result.result == "PASS":
                conversation_metrics[conv_id][metric_id]["pass"] += 1
            elif result.result == "FAIL":
                conversation_metrics[conv_id][metric_id]["fail"] += 1
            elif result.result == "ERROR":
                conversation_metrics[conv_id][metric_id]["error"] += 1

        if not conversation_metrics:
            return None

        # Get all unique conversations and metrics that actually have data
        all_conversations: List[str] = list(conversation_metrics.keys())
        all_metrics_set: set[str] = set()
        for conv_metrics in conversation_metrics.values():
            all_metrics_set.update(conv_metrics.keys())
        all_metrics: List[str] = sorted(all_metrics_set)

        # Build the heatmap matrix - only populate cells where data exists
        heatmap_data = []
        for conv_id in all_conversations:
            row = []
            for metric_id in all_metrics:
                if metric_id in conversation_metrics[conv_id]:
                    # Calculate pass rate for this metric in this conversation
                    stats = conversation_metrics[conv_id][metric_id]
                    total = stats["pass"] + stats["fail"] + stats["error"]
                    pass_rate = (stats["pass"] / total * 100) if total > 0 else 0
                    row.append(pass_rate)
                else:
                    # Use NaN for missing data (will show as blank/white in heatmap)
                    row.append(np.nan)
            heatmap_data.append(row)

        # Create DataFrame with explicit type conversion
        df = pd.DataFrame(
            data=heatmap_data,
            index=all_conversations,  # type: ignore
            columns=all_metrics,  # type: ignore
        )

        # Create heatmap
        _, ax = plt.subplots(figsize=tuple(self.figsize), dpi=self.dpi)

        sns.heatmap(
            df,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            cbar_kws={"label": "Pass Rate (%)"},
            ax=ax,
            mask=df.isna(),  # Mask NaN values (missing data)
            linewidths=0.5,
            linecolor="lightgray",
        )

        # Customize
        ax.set_title(
            "Conversation Performance Heatmap\n(Only showing metrics with data)",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Pass Rate (%)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Conversation Groups", fontsize=12, fontweight="bold")

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        plt.tight_layout()

        # Save
        filename = self.graphs_dir / f"{base_filename}_conversation_heatmap.png"
        plt.savefig(filename, dpi=self.dpi, bbox_inches="tight")
        plt.close()

        return filename

    def _calculate_detailed_summary_stats(
        self, results: List[EvaluationResult]
    ) -> Dict[str, Any]:
        """Calculate detailed summary statistics for graphs."""
        return calculate_detailed_stats(results)
