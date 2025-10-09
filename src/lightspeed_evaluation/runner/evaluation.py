"""LightSpeed Evaluation Framework - Main Evaluation Runner."""

import argparse
import sys
import traceback
from typing import Optional

# Import only lightweight modules at top level
from lightspeed_evaluation.core.constants import MAX_RUN_NAME_LENGTH
from lightspeed_evaluation.core.system import ConfigLoader


def run_evaluation(  # pylint: disable=too-many-locals
    system_config_path: str,
    evaluation_data_path: str,
    output_dir: Optional[str] = None,
    run_name: Optional[str] = None,
) -> Optional[dict[str, int]]:
    """Run the complete evaluation pipeline using EvaluationPipeline.

    Args:
        system_config_path: Path to system.yaml
        evaluation_data_path: Path to evaluation_data.yaml
        output_dir: Optional override for output directory
        run_name: Optional name for evaluation run (overrides YAML/filename default)

    Returns:
        dict: Summary statistics with keys TOTAL, PASS, FAIL, ERROR
    """
    print("🚀 LightSpeed Evaluation Framework")
    print("=" * 50)

    try:
        # Step 0: Setup environment from config
        print("🔧 Loading Configuration & Setting up environment and logging...")
        loader = ConfigLoader()
        system_config = loader.load_system_config(system_config_path)

        # pylint: disable=import-outside-toplevel

        # Step 1: Import heavy modules once environment & logging is set
        print("\n📋 Loading Heavy Modules...")
        from lightspeed_evaluation.core.output import OutputHandler
        from lightspeed_evaluation.core.output.statistics import calculate_basic_stats
        from lightspeed_evaluation.core.system import DataValidator
        from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

        # pylint: enable=import-outside-toplevel

        print("✅ Environment setup complete, modules loaded")

        llm_config = system_config.llm
        output_config = system_config.output

        # Step 2: Load and validate evaluation data
        data_validator = DataValidator(api_enabled=system_config.api.enabled)
        evaluation_data = data_validator.load_evaluation_data(
            evaluation_data_path, run_name_override=run_name
        )

        print(f"✅ System config: {llm_config.provider}/{llm_config.model}")
        print(f"✅ Evaluation data: {len(evaluation_data)} conversation groups")

        # Step 3: Run evaluation with pre-loaded data
        print("\n⚙️ Initializing Evaluation Pipeline...")
        pipeline = EvaluationPipeline(loader, output_dir)

        print("\n🔄 Running Evaluation...")
        try:
            results = pipeline.run_evaluation(evaluation_data, evaluation_data_path)
        finally:
            pipeline.close()

        # Step 4: Generate reports and calculate stats
        print("\n📊 Generating Reports...")
        output_handler = OutputHandler(
            output_dir=output_dir or output_config.output_dir,
            base_filename=output_config.base_filename,
            system_config=system_config,
        )

        # Generate reports based on configuration
        output_handler.generate_reports(results)

        print("\n🎉 Evaluation Complete!")
        print(f"📊 {len(results)} evaluations completed")
        print(f"📁 Reports generated in: {output_handler.output_dir}")

        # Step 5: Final Summary
        summary = calculate_basic_stats(results)
        print(
            f"✅ Pass: {summary['PASS']}, ❌ Fail: {summary['FAIL']}, ⚠️ Error: {summary['ERROR']}"
        )
        if summary["ERROR"] > 0:
            print(
                f"⚠️ {summary['ERROR']} evaluations had errors - check detailed report"
            )

        return {
            "TOTAL": summary["TOTAL"],
            "PASS": summary["PASS"],
            "FAIL": summary["FAIL"],
            "ERROR": summary["ERROR"],
        }

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n❌ Evaluation failed: {e}")
        traceback.print_exc()
        return None


def main() -> int:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="LightSpeed Evaluation Framework / Tool",
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
    parser.add_argument("--output-dir", help="Override output directory (optional)")
    parser.add_argument(
        "--run-name",
        help=f"Name for this evaluation run (overrides YAML value, max {MAX_RUN_NAME_LENGTH} chars)",
    )

    args = parser.parse_args()

    summary = run_evaluation(
        args.system_config, args.eval_data, args.output_dir, args.run_name
    )

    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())
