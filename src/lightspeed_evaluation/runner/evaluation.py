"""LightSpeed Evaluation Framework - Main Evaluation Runner."""

import argparse
import sys
import traceback
from typing import Any, Optional

# Import only lightweight modules at top level
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import DataValidationError


def _print_summary(
    summary: dict[str, Any],
    api_tokens: Optional[dict[str, int]] = None,
) -> None:
    """Print evaluation summary and token usage."""
    print(
        f"✅ Pass: {summary['PASS']}, ❌ Fail: {summary['FAIL']}, "
        f"⚠️ Error: {summary['ERROR']}, ⏭️ Skipped: {summary['SKIPPED']}"
    )
    if summary["ERROR"] > 0:
        print(f"⚠️ {summary['ERROR']} evaluations had errors - check detailed report")

    print("\n📊 Token Usage Summary:")
    print(
        f"Judge LLM: {summary['total_judge_llm_tokens']:,} tokens "
        f"(Input: {summary['total_judge_llm_input_tokens']:,}, "
        f"Output: {summary['total_judge_llm_output_tokens']:,})"
    )
    if api_tokens:
        print(
            f"API Calls: {api_tokens['total_api_tokens']:,} tokens "
            f"(Input: {api_tokens['total_api_input_tokens']:,}, "
            f"Output: {api_tokens['total_api_output_tokens']:,})"
        )
        total = summary["total_judge_llm_tokens"] + api_tokens["total_api_tokens"]
        print(f"Total: {total:,} tokens")


def run_evaluation(  # pylint: disable=too-many-locals
    eval_args: argparse.Namespace,
) -> Optional[dict[str, int]]:
    """Run the complete evaluation pipeline.

    Args:
        eval_args: Parsed command line arguments

    Returns:
        dict: Summary statistics with keys TOTAL, PASS, FAIL, ERROR, SKIPPED
    """
    print("🚀 LightSpeed Evaluation Framework")
    print("=" * 50)

    try:
        print("🔧 Loading Configuration & Setting up environment...")
        loader = ConfigLoader()
        system_config = loader.load_system_config(eval_args.system_config)

        # Import heavy modules after environment is configured
        print("\n📋 Loading Heavy Modules...")
        # pylint: disable=import-outside-toplevel
        from lightspeed_evaluation.core.output import OutputHandler
        from lightspeed_evaluation.core.output.statistics import (
            calculate_api_token_usage,
            calculate_basic_stats,
        )
        from lightspeed_evaluation.core.system import DataValidator
        from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

        # pylint: enable=import-outside-toplevel
        print("✅ Configuration loaded & Setup is done !")

        # Load and validate evaluation data
        evaluation_data = DataValidator(
            api_enabled=system_config.api.enabled,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
        ).load_evaluation_data(eval_args.eval_data)

        print(
            f"✅ System config: {system_config.llm.provider}/{system_config.llm.model}"
        )
        print(f"✅ Evaluation data: {len(evaluation_data)} conversation groups")

        # Run evaluation pipeline
        print("\n⚙️ Initializing Evaluation Pipeline...")
        pipeline = EvaluationPipeline(loader, eval_args.output_dir)

        print("\n🔄 Running Evaluation...")
        try:
            results = pipeline.run_evaluation(evaluation_data, eval_args.eval_data)
        finally:
            pipeline.close()

        # Generate reports
        print("\n📊 Generating Reports...")
        output_handler = OutputHandler(
            output_dir=eval_args.output_dir or system_config.output.output_dir,
            base_filename=system_config.output.base_filename,
            system_config=system_config,
        )
        output_handler.generate_reports(results, evaluation_data)

        print("\n🎉 Evaluation Complete!")
        print(f"📊 {len(results)} evaluations completed")
        print(f"📁 Reports generated in: {output_handler.output_dir}")

        # Final Summary
        summary = calculate_basic_stats(results)
        api_tokens = (
            calculate_api_token_usage(evaluation_data)
            if system_config.api.enabled
            else None
        )
        _print_summary(summary, api_tokens)

        return {
            "TOTAL": summary["TOTAL"],
            "PASS": summary["PASS"],
            "FAIL": summary["FAIL"],
            "ERROR": summary["ERROR"],
            "SKIPPED": summary["SKIPPED"],
        }

    except (FileNotFoundError, ValueError, RuntimeError, DataValidationError) as e:
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

    eval_args = parser.parse_args()

    summary = run_evaluation(eval_args)
    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())
