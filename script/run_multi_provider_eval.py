#!/usr/bin/env python3
"""Multi-Provider Evaluation Runner.

This script runs evaluations across multiple providers and models sequentially,
modifying the system configuration for each combination.
"""

import argparse
import copy
import logging
import os
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from lightspeed_evaluation.runner.evaluation import run_evaluation
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
            providers_config_path: Path to providers_config.yaml
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
        self.output_base = Path(self.settings.get("output_base", "./eval_output_multi_provider"))

        # Create output base directory
        self.output_base.mkdir(parents=True, exist_ok=True)

        # Track results
        self.results: list[dict[str, Any]] = []

    def _validate_paths(self) -> None:
        """Validate that required configuration files exist."""
        if not self.providers_config_path.exists():
            raise FileNotFoundError(
                f"Providers config not found: {self.providers_config_path}"
            )
        if not self.system_config_path.exists():
            raise FileNotFoundError(f"System config not found: {self.system_config_path}")
        if not self.eval_data_path.exists():
            raise FileNotFoundError(
                f"Evaluation data not found: {self.eval_data_path}"
            )

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        """Load a YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary containing the configuration
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file {path}: {e}") from e

    def _create_provider_model_configs(self) -> list[dict[str, Any]]:
        """Create a list of provider-model configurations to evaluate.

        Returns:
            List of dictionaries with provider, model, and settings
        """
        configs = []
        
        # Iterate through all keys except "settings"
        for provider_name, models in self.providers_config.items():
            # Skip the settings key
            if provider_name == "settings":
                continue
            
            # Skip if models is not a list
            if not isinstance(models, list):
                logger.warning(f"Skipping provider '{provider_name}': models is not a list")
                continue

            if not models:
                logger.warning(f"Skipping provider '{provider_name}': no models specified")
                continue

            # Use provider_name as provider_id
            for model in models:
                config = {
                    "provider_name": provider_name,
                    "provider_id": provider_name,
                    "model": model,
                    "settings": {},  # No additional settings in new format
                }
                configs.append(config)

        return configs

    def _create_modified_system_config(
        self, provider_id: str, model: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Create a modified system configuration for a specific provider-model.

        Args:
            provider_id: Provider identifier (e.g., "openai")
            model: Model name (e.g., "gpt-4o-mini")
            settings: Additional provider-specific settings

        Returns:
            Modified system configuration dictionary
        """
        # Deep copy the system config
        modified_config = copy.deepcopy(self.system_config)

        # Update LLM configuration
        if "llm" in modified_config:
            modified_config["llm"]["provider"] = provider_id
            modified_config["llm"]["model"] = model
            # Apply any provider-specific settings
            modified_config["llm"].update(settings)

        # Update API configuration if enabled
        if "api" in modified_config and modified_config["api"].get("enabled", False):
            modified_config["api"]["provider"] = provider_id
            modified_config["api"]["model"] = model

        return modified_config

    def _create_temp_system_config(
        self, provider_id: str, model: str, settings: dict[str, Any]
    ) -> Path:
        """Create a temporary system configuration file.

        Args:
            provider_id: Provider identifier
            model: Model name
            settings: Additional settings

        Returns:
            Path to the temporary configuration file
        """
        modified_config = self._create_modified_system_config(provider_id, model, settings)

        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"system_{provider_id}_{model.replace('/', '_')}_",
            delete=False,
            encoding="utf-8",
        )

        try:
            yaml.dump(modified_config, temp_file, default_flow_style=False, sort_keys=False)
            temp_file.flush()
            return Path(temp_file.name)
        finally:
            temp_file.close()

    def _run_single_evaluation(
        self, provider_name: str, provider_id: str, model: str, settings: dict[str, Any]
    ) -> dict[str, Any]:
        """Run a single evaluation for a provider-model combination.

        Args:
            provider_name: Human-readable provider name
            provider_id: Provider identifier
            model: Model name
            settings: Additional settings

        Returns:
            Dictionary containing evaluation results and metadata
        """
        start_time = datetime.now()
        temp_config_path: Optional[Path] = None
        
        # Sanitize model name for filesystem
        safe_model_name = model.replace("/", "_").replace(":", "_")
        
        # Create output directory for this provider-model
        output_dir = self.output_base / provider_id / safe_model_name
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
            temp_config_path = self._create_temp_system_config(
                provider_id, model, settings
            )

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
                result["success"] = True
                result["summary"] = summary
                logger.info(
                    f"✓ Completed: {provider_id}/{model} - "
                    f"Pass: {summary['PASS']}, Fail: {summary['FAIL']}, Error: {summary['ERROR']}"
                )
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
            logger.error("No valid provider-model configurations found!")
            return []

        logger.info(f"Running evaluations for {len(configs)} provider-model combinations")
        logger.info(f"Output base directory: {self.output_base}")
        logger.info("=" * 80)

        # Run evaluations sequentially
        for config in configs:
            result = self._run_single_evaluation(
                config["provider_name"],
                config["provider_id"],
                config["model"],
                config["settings"],
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
                print(f"  Results: Pass: {eval_summary['PASS']}, "
                      f"Fail: {eval_summary['FAIL']}, "
                      f"Error: {eval_summary['ERROR']}")
            
            if result["error"]:
                print(f"  Error: {result['error']}")

        print("\n" + "=" * 80)


def main() -> int:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run evaluations across multiple providers and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluations with default config locations
  python3 run_multi_provider_eval.py --providers-config config/providers_config.yaml
  
  # Run with custom configurations
  python3 run_multi_provider_eval.py \\
      --providers-config config/providers_config.yaml \\
      --system-config config/system.yaml \\
      --eval-data config/evaluation_data.yaml
  
  # Run with verbose output
  python3 run_multi_provider_eval.py \\
      --providers-config config/providers_config.yaml \\
      --verbose
        """,
    )

    parser.add_argument(
        "--providers-config",
        default="config/providers_config.yaml",
        help="Path to providers configuration file (providers_config.yaml)",
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

        # Exit with appropriate code
        exit_code = 0 if summary["failed"] == 0 else 1
        
        # Force exit to prevent hanging threads from underlying libraries
        logger.info("Evaluation complete. Exiting...")
        os._exit(exit_code)

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        os._exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        os._exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            traceback.print_exc()
        os._exit(1)


if __name__ == "__main__":
    main()

