"""
LSC Evaluation Framework - Main Evaluation Runner.

Simple interface that uses EvaluationEngine as the core controller.

Usage:
    python -m runner --system-config config/system.yaml --eval-data config/evaluation_data.yaml

Or programmatically:
    from runner import run_evaluation
    results = run_evaluation("config/system.yaml", "config/evaluation_data.yaml")
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional

from lsc_eval import ConfigLoader, DataValidator, EvaluationEngine, OutputHandler
from lsc_eval.core import setup_environment_variables
from lsc_eval.output.utils import calculate_basic_stats


def run_evaluation(
    system_config_path: str, evaluation_data_path: str, output_dir: Optional[str] = None
) -> Optional[Dict[str, int]]:
    """
    Run the complete evaluation pipeline using EvaluationEngine.

    Args:
        system_config_path: Path to system.yaml
        evaluation_data_path: Path to evaluation_data.yaml
        output_dir: Optional override for output directory

    Returns:
        dict: Summary statistics with keys TOTAL, PASS, FAIL, ERROR
    """
    print("🚀 LSC Evaluation Framework")
    print("=" * 50)

    try:
        # Step 1: Load configuration
        print("\n📋 Loading Configuration...")
        loader = ConfigLoader()
        system_config = loader.load_system_config(system_config_path)

        data_validator = DataValidator()
        evaluation_data = data_validator.load_evaluation_data(evaluation_data_path)

        print(
            f"✅ System config: {system_config.llm_provider}/{system_config.llm_model}"
        )
        print(f"✅ Evaluation data: {len(evaluation_data)} conversation groups")

        # Step 2: Initialize evaluation engine (core controller)
        print("\n⚙️ Initializing Evaluation Engine...")
        engine = EvaluationEngine(loader)

        # Step 3: Run evaluation (engine controls the flow)
        print("\n🔄 Running Evaluation...")
        results = engine.run_evaluation(evaluation_data)

        # Step 4: Generate reports
        print("\n📊 Generating Reports...")
        output_handler = OutputHandler(
            output_dir=output_dir or system_config.output_dir,
            base_filename=system_config.base_filename,
            system_config=system_config,
        )

        output_handler.generate_reports(
            results, include_graphs=system_config.include_graphs
        )

        print("\n🎉 Evaluation Complete!")
        print(f"📊 {len(results)} evaluations completed")
        print(f"📁 Reports generated in: {output_handler.output_dir}")

        # Calculate and show summary
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
    parser = argparse.ArgumentParser(description="LSC Evaluation Framework / Tool")
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

    args = parser.parse_args()

    # CRITICAL: Setup environment variables from system config FIRST
    setup_environment_variables(args.system_config)

    # Validate input files exist
    if not Path(args.system_config).exists():
        print(f"❌ System config file not found: {args.system_config}")
        return 1

    if not Path(args.eval_data).exists():
        print(f"❌ Evaluation data file not found: {args.eval_data}")
        return 1

    # Run evaluation
    summary = run_evaluation(args.system_config, args.eval_data, args.output_dir)

    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())
