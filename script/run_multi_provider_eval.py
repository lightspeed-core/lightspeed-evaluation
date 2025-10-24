#!/usr/bin/env python3
"""Multi-Provider Evaluation Runner with Statistical Analysis.

This script runs evaluations across multiple providers and models sequentially,
modifying the system configuration for each combination, and then performs
comprehensive statistical analysis to determine the best model.
"""

import argparse
import copy
import json
import re
import logging
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from lightspeed_evaluation.runner.evaluation import run_evaluation
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class MultiProviderEvaluationRunner:
    """Runner for executing evaluations across multiple providers and models."""

    def __init__(
        self,
        providers_config_path: str,
        system_config_path: str,
        eval_data_path: str,
    ):
        """Initialize the multi-provider evaluation runner.

        Args:
            providers_config_path: Path to multi_eval_config.yaml
            system_config_path: Path to system.yaml
            eval_data_path: Path to evaluation_data.yaml
        """
        self.providers_config_path = Path(providers_config_path)
        self.system_config_path = Path(system_config_path)
        self.eval_data_path = Path(eval_data_path)

        # Validate paths
        self._validate_paths()

        # Load configurations
        self.providers_config = self._load_yaml(self.providers_config_path)
        self.system_config = self._load_yaml(self.system_config_path)

        # Extract settings
        self.settings = self.providers_config.get("settings", {})
        self.output_base = Path(
            self.settings.get("output_base", "./eval_output_multi_provider")
        )

        # Create output base directory
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Track results
        self.results: list[dict[str, Any]] = []

        # Track model analysis
        self.model_stats: dict[str, dict[str, Any]] = {}

    def _validate_paths(self) -> None:
        """Validate that required configuration files exist."""
        if not self.providers_config_path.exists():
            raise FileNotFoundError(
                f"Providers config not found: {self.providers_config_path}"
            )
        if not self.system_config_path.exists():
            raise FileNotFoundError(
                f"System config not found: {self.system_config_path}"
            )
        if not self.eval_data_path.exists():
            raise FileNotFoundError(f"Evaluation data not found: {self.eval_data_path}")

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load a YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary containing the configuration
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if data is None:
                    return {}
                if not isinstance(data, dict):
                    raise ValueError(
                        f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}"
                    )
                return data
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {path}: {e}") from e

    def _create_provider_model_configs(self) -> list[dict[str, Any]]:
        """Create a list of provider-model configurations to evaluate.

        Returns:
            List of dictionaries with provider, model, and settings
        """
        configs = []

        # Get providers from the config
        providers = self.providers_config.get("providers", {})

        if not providers:
            logger.warning("No 'providers' section found in configuration")
            return configs

        # Iterate through all providers
        for provider_name, provider_config in providers.items():
            # Handle both old format (list) and new format (dict with models)
            models = []

            if isinstance(provider_config, list):
                # Old format: direct list of models
                models = provider_config
            elif isinstance(provider_config, dict):
                # New format: dict with 'models'
                models = provider_config.get("models", [])
            else:
                logger.warning(
                    f"Skipping provider '{provider_name}': invalid configuration format"
                )
                continue

            if not models:
                logger.warning(
                    f"Skipping provider '{provider_name}': no models specified"
                )
                continue

            # Use provider_name as provider_id
            for model in models:
                config = {
                    "provider_name": provider_name,
                    "provider_id": provider_name,
                    "model": model,
                }
                configs.append(config)

        return configs

    def _create_modified_system_config(
        self, provider_id: str, model: str
    ) -> dict[str, Any]:
        """Create a modified system configuration for a specific provider-model.

        Args:
            provider_id: Provider identifier (e.g., "openai")
            model: Model name (e.g., "gpt-4o-mini")

        Returns:
            Modified system configuration dictionary
        """
        # Deep copy the system config
        modified_config = copy.deepcopy(self.system_config)

        # Update API configuration if enabled
        if "api" in modified_config and modified_config["api"].get("enabled", False):
            modified_config["api"]["provider"] = provider_id
            modified_config["api"]["model"] = model

        return modified_config

    def _create_temp_system_config(self, provider_id: str, model: str) -> Path:
        """Create a temporary system configuration file.

        Args:
            provider_id: Provider identifier
            model: Model name

        Returns:
            Path to the temporary configuration file
        """
        modified_config = self._create_modified_system_config(provider_id, model)

        temp_path = None
        # Sanitize names for safe filesystem usage
        safe_provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", provider_id).strip("._")
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("._")
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"system_{safe_provider}_{safe_model}_",
            delete=False,
            encoding="utf-8",
        )
        temp_path = Path(temp_file.name)

        try:
            yaml.dump(
                modified_config, temp_file, default_flow_style=False, sort_keys=False
            )
            temp_file.flush()
            return temp_path
        except Exception:
            # Clean up the temp file if dump/flush fails
            if temp_path and temp_path.exists():
                temp_path.unlink()
            raise
        finally:
            temp_file.close()

    def _run_single_evaluation(
        self, provider_name: str, provider_id: str, model: str
    ) -> dict[str, Any]:
        """Run a single evaluation for a provider-model combination.

        Args:
            provider_name: Human-readable provider name
            provider_id: Provider identifier
            model: Model name

        Returns:
            Dictionary containing evaluation results and metadata
        """
        start_time = datetime.now()
        temp_config_path: Optional[Path] = None

        # Sanitize names for filesystem and enforce confinement under output_base
        safe_provider = re.sub(r"[^A-Za-z0-9_.-]+", "_", provider_id).strip("._")
        safe_model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("._")

        # Create output directory for this provider-model with path traversal protection
        base = self.output_base.resolve()
        output_dir = (base / safe_provider / safe_model_name).resolve()
        if not output_dir.is_relative_to(base):
            raise ValueError(f"Unsafe provider/model path: {provider_id}/{model}")
        output_dir.mkdir(parents=True, exist_ok=True)

        result = {
            "provider_name": provider_name,
            "provider_id": provider_id,
            "model": model,
            "output_dir": str(output_dir),
            "start_time": start_time.isoformat(),
            "success": False,
            "error": None,
        }

        try:
            logger.info(f"Starting evaluation: {provider_id}/{model}")

            # Create temporary system config
            temp_config_path = self._create_temp_system_config(provider_id, model)

            logger.debug(f"Using temp config: {temp_config_path}")
            logger.debug(f"Evaluation data: {self.eval_data_path}")
            logger.debug(f"Output directory: {output_dir}")

            # Run evaluation by calling the function directly
            summary = run_evaluation(
                system_config_path=str(temp_config_path),
                evaluation_data_path=str(self.eval_data_path),
                output_dir=str(output_dir),
            )

            # Check result
            if summary is not None:
                # Validate expected keys exist
                if all(k in summary for k in ("PASS", "FAIL", "ERROR")):
                    result["success"] = True
                    result["summary"] = summary
                    logger.info(
                        f"✓ Completed: {provider_id}/{model} - "
                        f"Pass: {summary['PASS']}, Fail: {summary['FAIL']}, Error: {summary['ERROR']}"
                    )
                else:
                    result["error"] = f"Invalid summary structure: {summary}"
                    logger.error(f"✗ Failed: {provider_id}/{model} - {result['error']}")
            else:
                result["error"] = "Evaluation returned None (failed)"
                logger.error(f"✗ Failed: {provider_id}/{model} - {result['error']}")

        except Exception as e:  # pylint: disable=broad-except
            result["error"] = f"Exception: {str(e)}"
            logger.error(f"✗ Exception in {provider_id}/{model}: {e}")
            logger.debug(traceback.format_exc())

        finally:
            # Clean up temporary config file
            if temp_config_path and temp_config_path.exists():
                try:
                    temp_config_path.unlink()
                except OSError:
                    logger.warning(f"Failed to delete temp config: {temp_config_path}")

        # Record end time and duration
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration_seconds"] = (end_time - start_time).total_seconds()

        return result

    def run_evaluations(self) -> list[dict[str, Any]]:
        """Run evaluations for all provider-model combinations sequentially.

        Returns:
            List of result dictionaries for each evaluation
        """
        # Get all provider-model configurations
        configs = self._create_provider_model_configs()

        if not configs:
            msg = "No valid provider-model configurations found (check multi_eval_config.yaml)"
            logger.error(msg)
            raise ValueError(msg)

        logger.info(
            f"Running evaluations for {len(configs)} provider-model combinations"
        )
        logger.info(f"Output base directory: {self.output_base}")
        logger.info("=" * 80)

        # Run evaluations sequentially
        for config in configs:
            result = self._run_single_evaluation(
                config["provider_name"],
                config["provider_id"],
                config["model"],
            )
            self.results.append(result)

        return self.results

    def generate_summary(self) -> dict[str, Any]:
        """Generate a summary of all evaluation runs.

        Returns:
            Dictionary containing summary statistics and results
        """
        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%",
            "output_base": str(self.output_base),
            "results": self.results,
        }

        return summary

    def save_summary(self, summary: dict[str, Any]) -> Path:
        """Save the evaluation summary to a file.

        Args:
            summary: Summary dictionary

        Returns:
            Path to the saved summary file
        """
        summary_path = self.output_base / "multi_provider_evaluation_summary.yaml"

        with open(summary_path, "w", encoding="utf-8") as f:
            yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Summary saved to: {summary_path}")
        return summary_path

    def print_summary(self, summary: dict[str, Any]) -> None:
        """Print a human-readable summary of the evaluation runs.

        Args:
            summary: Summary dictionary
        """
        print("\n" + "=" * 80)
        print("MULTI-PROVIDER EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal Evaluations: {summary['total_evaluations']}")
        print(f"Successful: {summary['successful']} ✓")
        print(f"Failed: {summary['failed']} ✗")
        print(f"Success Rate: {summary['success_rate']}")
        print(f"\nOutput Directory: {summary['output_base']}")
        print("\n" + "-" * 80)
        print("Individual Results:")
        print("-" * 80)

        for result in summary["results"]:
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            duration = result.get("duration_seconds", 0)
            print(f"\n{result['provider_id']}/{result['model']}:")
            print(f"  Status: {status}")
            print(f"  Duration: {duration:.1f}s")
            print(f"  Output: {result['output_dir']}")

            # Show evaluation summary if available
            if result.get("summary"):
                eval_summary = result["summary"]
                print(
                    f"  Results: Pass: {eval_summary['PASS']}, "
                    f"Fail: {eval_summary['FAIL']}, "
                    f"Error: {eval_summary['ERROR']}"
                )

            if result["error"]:
                print(f"  Error: {result['error']}")

        print("\n" + "=" * 80)

    def analyze_model_performance(self) -> None:
        """Analyze performance of all evaluated models."""
        logger.info("\n" + "=" * 80)
        logger.info("Analyzing model performance...")
        logger.info("=" * 80)

        # Load evaluation summaries for successful runs
        for result in self.results:
            if not result["success"]:
                continue

            model_key = f"{result['provider_id']}/{result['model']}"
            output_dir = Path(result["output_dir"])

            # Find summary JSON file
            summary_files = list(output_dir.glob("*_summary.json"))
            if not summary_files:
                logger.warning(f"No summary file found for {model_key}")
                continue

            # Load the most recent summary file
            summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    summary_data = json.load(f)

                self.model_stats[model_key] = self._analyze_single_model(
                    model_key, summary_data
                )
            except Exception as e:
                logger.error(f"Failed to analyze {model_key}: {e}")

    def _analyze_single_model(
        self, model_key: str, summary_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze a single model's performance.

        Args:
            model_key: Model identifier (e.g., "openai/gpt-4o-mini")
            summary_data: Summary data from JSON file

        Returns:
            Dictionary containing analysis results
        """
        results = summary_data.get("results", [])
        summary_stats = summary_data.get("summary_stats", {})
        overall = summary_stats.get("overall", {})

        # Extract basic metrics
        total = overall.get("TOTAL", 0)
        passed = overall.get("PASS", 0)
        failed = overall.get("FAIL", 0)
        errors = overall.get("ERROR", 0)

        # Calculate rates (convert from percentage if needed)
        pass_rate_raw = overall.get("pass_rate", 0)
        fail_rate_raw = overall.get("fail_rate", 0)
        error_rate_raw = overall.get("error_rate", 0)

        # Check if rates are in percentage format (>1.0) or decimal format (0.0-1.0)
        if pass_rate_raw > 1.0 or fail_rate_raw > 1.0 or error_rate_raw > 1.0:
            # Rates are percentages, convert to decimal
            pass_rate = pass_rate_raw / 100.0
            fail_rate = fail_rate_raw / 100.0
            error_rate = error_rate_raw / 100.0
        else:
            # Rates are already in decimal format
            pass_rate = pass_rate_raw
            fail_rate = fail_rate_raw
            error_rate = error_rate_raw

        # Calculate success rate (non-error rate)
        success_rate = (passed + failed) / total if total > 0 else 0

        # Collect all scores
        all_scores = [float(r["score"]) for r in results if r.get("score") is not None]

        # Calculate score statistics
        if all_scores:
            score_stats = {
                "mean": float(np.mean(all_scores)),
                "median": float(np.median(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores)),
                "count": len(all_scores),
            }

            # Calculate confidence interval for mean score (95% confidence)
            if len(all_scores) > 1:
                try:
                    from scipy import stats as scipy_stats

                    confidence_level = 0.95
                    degrees_freedom = len(all_scores) - 1
                    sample_std = float(np.std(all_scores, ddof=1))
                    std_error = sample_std / np.sqrt(len(all_scores))
                    # Using t-distribution critical value
                    t_critical = scipy_stats.t.ppf(
                        (1 + confidence_level) / 2, degrees_freedom
                    )
                    margin_of_error = t_critical * std_error
                    score_stats["confidence_interval"] = {
                        "low": float(
                            max(0.0, score_stats["mean"] - margin_of_error)
                        ),  # Clamp to 0
                        "high": float(
                            min(1.0, score_stats["mean"] + margin_of_error)
                        ),  # Clamp to 1
                        "mean": float(score_stats["mean"]),
                        "confidence_level": confidence_level * 100,
                    }
                except ImportError:
                    logger.warning(
                        "scipy not available, skipping confidence interval calculation"
                    )
                    score_stats["confidence_interval"] = None
            else:
                # Single score - no confidence interval
                score_stats["confidence_interval"] = None
        else:
            score_stats = {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
                "confidence_interval": None,
            }

        # Get metric-level statistics from summary_stats if available
        by_metric_stats = summary_stats.get("by_metric", {})
        metric_stats = {}

        for metric_name, metric_data in by_metric_stats.items():
            # Extract data from summary
            pass_count = metric_data.get("pass", 0)
            fail_count = metric_data.get("fail", 0)
            error_count = metric_data.get("error", 0)
            total_metric = pass_count + fail_count + error_count

            # Get pass/fail/error rates (convert from percentage if needed)
            pass_rate_metric = metric_data.get("pass_rate", 0)
            fail_rate_metric = metric_data.get("fail_rate", 0)
            error_rate_metric = metric_data.get("error_rate", 0)

            # Convert percentages to decimals if needed
            if pass_rate_metric > 1.0 or fail_rate_metric > 1.0:
                pass_rate_metric = pass_rate_metric / 100.0
                fail_rate_metric = fail_rate_metric / 100.0
                error_rate_metric = error_rate_metric / 100.0

            # Get score statistics
            score_stats_metric = metric_data.get("score_statistics", {})
            mean_score = score_stats_metric.get("mean", 0.0)

            metric_stats[metric_name] = {
                "mean_score": float(mean_score),
                "pass_rate": float(pass_rate_metric),
                "fail_rate": float(fail_rate_metric),
                "error_rate": float(error_rate_metric),
                "total_evaluations": total_metric,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "error_count": error_count,
            }

        # Calculate composite score (weighted combination of metrics)
        composite_score = self._calculate_composite_score(
            pass_rate, error_rate, score_stats["mean"], success_rate
        )

        return {
            "model_key": model_key,
            "overall": {
                "total_evaluations": total,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": pass_rate,
                "fail_rate": fail_rate,
                "error_rate": error_rate,
                "success_rate": success_rate,
            },
            "score_statistics": score_stats,
            "metric_statistics": metric_stats,
            "composite_score": composite_score,
        }

    def _calculate_composite_score(
        self,
        pass_rate: float,
        error_rate: float,
        mean_score: float,
        success_rate: float,
    ) -> float:
        """Calculate a composite score for ranking models.

        The composite score is a weighted combination of:
        - Pass rate (40% weight)
        - Mean score (30% weight)
        - Success rate/non-error rate (20% weight)
        - Inverse error rate penalty (10% weight)

        Args:
            pass_rate: Proportion of passed evaluations
            error_rate: Proportion of errored evaluations
            mean_score: Mean score across all evaluations
            success_rate: Proportion of non-error evaluations

        Returns:
            Composite score between 0 and 1 (higher is better)
        """
        PASS_RATE_WEIGHT = 0.40
        MEAN_SCORE_WEIGHT = 0.30
        SUCCESS_RATE_WEIGHT = 0.20
        ERROR_PENALTY_WEIGHT = 0.10

        composite = (
            (pass_rate * PASS_RATE_WEIGHT)
            + (mean_score * MEAN_SCORE_WEIGHT)
            + (success_rate * SUCCESS_RATE_WEIGHT)
            + ((1 - error_rate) * ERROR_PENALTY_WEIGHT)
        )

        return composite

    def rank_models(self) -> list[tuple[str, dict[str, Any]]]:
        """Rank models by composite score.

        Returns:
            List of (model_key, stats) tuples sorted by composite score (best first)
        """
        ranked = sorted(
            self.model_stats.items(),
            key=lambda x: x[1]["composite_score"],
            reverse=True,
        )
        return ranked

    def print_statistical_comparison(self) -> None:
        """Print comprehensive statistical comparison of all models."""
        if not self.model_stats:
            logger.warning("No model statistics available for comparison")
            return

        ranked = self.rank_models()

        print("\n" + "=" * 80)
        print("STATISTICAL MODEL COMPARISON & BEST MODEL ANALYSIS")
        print("=" * 80)
        print(f"\nTotal Models Analyzed: {len(self.model_stats)}")

        # Best model section
        if ranked:
            best_model_key, best_stats = ranked[0]
            print("\n" + "=" * 80)
            print("🏆 BEST MODEL")
            print("=" * 80)
            print(f"\nModel: {best_model_key}")
            print(
                f"Composite Score: {best_stats['composite_score']:.4f} (higher is better)"
            )
            print("\nPerformance Summary:")
            print(f"  ✅ Pass Rate: {best_stats['overall']['pass_rate'] * 100:.2f}%")
            print(f"  ❌ Fail Rate: {best_stats['overall']['fail_rate'] * 100:.2f}%")
            print(f"  ⚠️  Error Rate: {best_stats['overall']['error_rate'] * 100:.2f}%")
            print(f"  📊 Mean Score: {best_stats['score_statistics']['mean']:.4f}")

            # Show confidence interval if available
            ci = best_stats["score_statistics"].get("confidence_interval")
            if ci:
                print(
                    f"  📈 95% Confidence Interval: [{ci['low']:.4f}, {ci['high']:.4f}]"
                )

            print(
                f"  🎯 Success Rate (Non-Error): {best_stats['overall']['success_rate'] * 100:.2f}%"
            )
            print(
                f"  🔢 Total Evaluations: {best_stats['overall']['total_evaluations']}"
            )

            print("\nWhy This Model is Best:")
            print("  • Highest composite score combining multiple metrics")
            print(
                f"  • Pass rate: {best_stats['overall']['passed']}/{best_stats['overall']['total_evaluations']}"
            )
            print(
                f"  • Low error rate: {best_stats['overall']['error_rate'] * 100:.2f}%"
            )
            print(
                f"  • Consistent performance: σ = {best_stats['score_statistics']['std']:.4f}"
            )

        # Full rankings with statistical comparison
        print("\n" + "=" * 80)
        print("COMPLETE MODEL RANKINGS")
        print("=" * 80)

        for rank, (model_key, stats) in enumerate(ranked, 1):
            print(f"\n{rank}. {model_key}")
            print(f"   Composite Score: {stats['composite_score']:.4f}")
            print(
                f"   Pass Rate: {stats['overall']['pass_rate'] * 100:.2f}% | "
                f"Mean Score: {stats['score_statistics']['mean']:.4f} | "
                f"Error Rate: {stats['overall']['error_rate'] * 100:.2f}%"
            )

            # Show confidence interval
            ci = stats["score_statistics"].get("confidence_interval")
            if ci:
                print(f"   95% CI: [{ci['low']:.4f}, {ci['high']:.4f}]")

        # Confidence interval comparison
        if len(ranked) >= 2:
            print("\n" + "=" * 80)
            print("STATISTICAL SIGNIFICANCE ANALYSIS")
            print("=" * 80)

            best_model_key, best_stats = ranked[0]
            best_ci = best_stats["score_statistics"].get("confidence_interval")

            if best_ci:
                print(f"\nBest Model ({best_model_key}) Confidence Interval:")
                print(f"  95% CI: [{best_ci['low']:.4f}, {best_ci['high']:.4f}]")
                print(f"  Mean: {best_stats['score_statistics']['mean']:.4f}")

                print("\nComparison with Other Models:")
                for rank, (model_key, stats) in enumerate(ranked[1:], 2):
                    ci = stats["score_statistics"].get("confidence_interval")
                    if ci:
                        # Check if confidence intervals overlap
                        intervals_overlap = max(best_ci["low"], ci["low"]) <= min(
                            best_ci["high"], ci["high"]
                        )

                        print(f"\n  {rank}. {model_key}:")
                        print(f"     95% CI: [{ci['low']:.4f}, {ci['high']:.4f}]")
                        print(f"     Mean: {stats['score_statistics']['mean']:.4f}")

                        if intervals_overlap:
                            print("     ⚠️  Confidence intervals OVERLAP")
                            print(
                                "     → No statistically significant difference at 95% confidence"
                            )
                            print(
                                "     → Consider both models (factors: cost, latency, specific use case)"
                            )
                        else:
                            print("     ✅ Confidence intervals DO NOT overlap")
                            print(
                                "     → Statistically significant difference at 95% confidence"
                            )
                            print(f"     → {best_model_key} is significantly better")

        # Detailed statistics
        print("\n" + "=" * 80)
        print("DETAILED STATISTICS BY MODEL")
        print("=" * 80)

        for model_key, stats in ranked:
            print(f"\n{model_key}")
            print("-" * 80)

            # Overall metrics
            print("Overall Performance:")
            print(f"  Total Evaluations: {stats['overall']['total_evaluations']}")
            print(
                f"  Pass: {stats['overall']['passed']} ({stats['overall']['pass_rate'] * 100:.2f}%)"
            )
            print(
                f"  Fail: {stats['overall']['failed']} ({stats['overall']['fail_rate'] * 100:.2f}%)"
            )
            print(
                f"  Error: {stats['overall']['errors']} ({stats['overall']['error_rate'] * 100:.2f}%)"
            )

            # Score statistics
            score_stats = stats["score_statistics"]
            print("\nScore Statistics:")
            print(f"  Mean: {score_stats['mean']:.4f}")
            print(f"  Median: {score_stats['median']:.4f}")
            print(f"  Std Dev: {score_stats['std']:.4f}")
            print(f"  Range: [{score_stats['min']:.4f}, {score_stats['max']:.4f}]")

            # Metric-level performance
            if stats["metric_statistics"]:
                print("\nPerformance by Metric:")
                for metric_name, metric_stats in stats["metric_statistics"].items():
                    print(f"  {metric_name}:")
                    print(f"    Mean Score: {metric_stats['mean_score']:.4f}")
                    print(
                        f"    Pass: {metric_stats['pass_count']}, "
                        f"Fail: {metric_stats['fail_count']}, "
                        f"Error: {metric_stats['error_count']}"
                    )
                    print(
                        f"    Pass Rate: {metric_stats['pass_rate'] * 100:.2f}% | "
                        f"Fail Rate: {metric_stats['fail_rate'] * 100:.2f}% | "
                        f"Error Rate: {metric_stats['error_rate'] * 100:.2f}%"
                    )

        # Recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        if ranked:
            best_model_key, best_stats = ranked[0]
            print(f"\n✅ RECOMMENDED MODEL: {best_model_key}")
            print(
                f"\nThis model achieved the highest composite score of {best_stats['composite_score']:.4f}"
            )
            print("considering pass rate, mean score, error rate, and consistency.")

            # Provide context if there are close contenders
            if len(ranked) > 1:
                second_best_key, second_stats = ranked[1]
                score_diff = (
                    best_stats["composite_score"] - second_stats["composite_score"]
                )

                if score_diff < 0.05:  # Less than 5% difference
                    print(
                        f"\n⚠️  NOTE: {second_best_key} is a close second with a score of "
                        f"{second_stats['composite_score']:.4f}"
                    )
                    print("   Consider both models based on:")
                    print("   • Cost per token")
                    print("   • Response latency")
                    print("   • Specific metric requirements")
                    print("   • API availability and rate limits")

        print("\n" + "=" * 80)

    def save_model_comparison(self) -> Path:
        """Save the model comparison analysis to a file.

        Returns:
            Path to the saved file
        """
        if not self.model_stats:
            logger.warning("No model statistics to save")
            return self.output_base

        output_path = self.output_base / "model_comparison_analysis.yaml"

        # Prepare data for output
        ranked = self.rank_models()
        analysis_data = {
            "total_models": len(self.model_stats),
            "output_base": str(self.output_base),
            "timestamp": datetime.now().isoformat(),
            "rankings": [
                {
                    "rank": rank,
                    "model": model_key,
                    "composite_score": stats["composite_score"],
                    "statistics": stats,
                }
                for rank, (model_key, stats) in enumerate(ranked, 1)
            ],
            "best_model": None,
        }

        if ranked:
            model_key, stats = ranked[0]
            analysis_data["best_model"] = {
                "model": model_key,
                "composite_score": stats["composite_score"],
                "pass_rate": stats["overall"]["pass_rate"],
                "mean_score": stats["score_statistics"]["mean"],
                "error_rate": stats["overall"]["error_rate"],
                "confidence_interval": stats["score_statistics"].get(
                    "confidence_interval"
                ),
            }

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(analysis_data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Model comparison analysis saved to: {output_path}")
        return output_path


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run evaluations across multiple providers and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluations with default config locations
  python3 run_multi_provider_eval.py --providers-config config/multi_eval_config.yaml
  
  # Run with custom configurations
  python3 run_multi_provider_eval.py \\
      --providers-config config/multi_eval_config.yaml \\
      --system-config config/system.yaml \\
      --eval-data config/evaluation_data.yaml
  
  # Run with verbose output
  python3 run_multi_provider_eval.py \\
      --providers-config config/multi_eval_config.yaml \\
      --verbose
        """,
    )

    parser.add_argument(
        "--providers-config",
        default="config/multi_eval_config.yaml",
        help="Path to multi-evaluation configuration file (multi_eval_config.yaml)",
    )
    parser.add_argument(
        "--system-config",
        default="config/system.yaml",
        help="Path to system configuration file (default: config/system.yaml)",
    )
    parser.add_argument(
        "--eval-data",
        default="config/evaluation_data.yaml",
        help="Path to evaluation data file (default: config/evaluation_data.yaml)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize runner
        runner = MultiProviderEvaluationRunner(
            providers_config_path=args.providers_config,
            system_config_path=args.system_config,
            eval_data_path=args.eval_data,
        )

        # Run evaluations
        runner.run_evaluations()

        # Generate and save summary
        summary = runner.generate_summary()
        runner.save_summary(summary)
        runner.print_summary(summary)

        # Analyze model performance and determine best model
        if summary["successful"] > 0:
            runner.analyze_model_performance()
            runner.print_statistical_comparison()
            runner.save_model_comparison()
        else:
            logger.warning("No successful evaluations to analyze")

        # Exit with appropriate code
        exit_code = 0 if summary["failed"] == 0 else 1
        logger.info("Evaluation complete. Exiting...")
        return exit_code

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
