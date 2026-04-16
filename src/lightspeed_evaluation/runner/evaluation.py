"""LightSpeed Evaluation Framework - Main Evaluation Runner."""

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

from lightspeed_evaluation.core.models.system import LLMPoolConfig, SystemConfig

# Import only lightweight modules at top level
from lightspeed_evaluation.core.storage import FileBackendConfig, get_file_config
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import (
    ConfigurationError,
    DataValidationError,
    StorageError,
)


def _clear_caches(system_config: SystemConfig) -> None:
    """Clear all cache directories for warmup mode.

    Args:
        system_config: System configuration containing cache directory paths
    """
    cache_dirs: list[tuple[str, str]] = []

    # Collect all enabled cache directories
    pool = system_config.llm_pool
    if isinstance(pool, LLMPoolConfig) and pool.defaults.cache_enabled:
        cache_dirs.append(("LLM Judge (pool)", pool.defaults.cache_dir))
    if system_config.llm.cache_enabled:
        cache_dirs.append(("LLM Judge", system_config.llm.cache_dir))
    # We clear the api cache even if the Lightspeed core api is disabled
    if system_config.api.cache_enabled:
        cache_dirs.append(("API", system_config.api.cache_dir))
    if system_config.embedding.cache_enabled:
        cache_dirs.append(("Embedding", system_config.embedding.cache_dir))

    if not cache_dirs:
        print("   No caches enabled to clear")
        return

    # Clear each cache directory
    for cache_name, cache_dir in cache_dirs:
        path = Path(cache_dir)
        resolved_path = path.resolve()
        if resolved_path in {Path("/"), Path.cwd()}:
            raise DataValidationError(
                f"Refusing to delete unsafe cache directory: '{resolved_path}'"
            )
        if path.exists():
            shutil.rmtree(path)
            print(f"   Cleared {cache_name} cache: {cache_dir}")
        # Recreate empty directory
        path.mkdir(parents=True, exist_ok=True)


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

        # Clear caches if cache warmup mode is enabled
        if eval_args.cache_warmup:
            print("\n🔥 Cache warmup mode: Clearing existing caches...")
            _clear_caches(system_config)

        # Import heavy modules after environment is configured
        print("\n📋 Loading Heavy Modules...")
        # pylint: disable=import-outside-toplevel
        from lightspeed_evaluation.api import evaluate
        from lightspeed_evaluation.core.output import OutputHandler
        from lightspeed_evaluation.core.output.statistics import (
            calculate_api_token_usage,
            calculate_basic_stats,
        )
        from lightspeed_evaluation.core.system import DataValidator

        # pylint: enable=import-outside-toplevel
        print("✅ Configuration loaded & Setup is done !")

        # Load, filter, and validate evaluation data
        evaluation_data = DataValidator(
            api_enabled=system_config.api.enabled,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
            system_config=system_config,
        ).load_evaluation_data(
            eval_args.eval_data,
            tags=eval_args.tags,
            conv_ids=eval_args.conv_ids,
        )

        print(
            f"✅ System config: {system_config.llm.provider}/{system_config.llm.model}"
        )

        # Handle case where no conversations match the filter
        if len(evaluation_data) == 0:
            print("\n⚠️ No conversation groups matched the filter criteria")
            print("   Nothing to evaluate - returning empty results")
            return {"TOTAL": 0, "PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}

        # Run evaluation pipeline
        print("\n⚙️ Initializing Evaluation Pipeline...")

        print("\n🔄 Running Evaluation...")
        results = evaluate(
            system_config, evaluation_data, output_dir=eval_args.output_dir
        )

        file_entries = [
            c for c in system_config.storage if isinstance(c, FileBackendConfig)
        ]
        if not file_entries:
            # No file storage in config: use legacy default file settings (same as get_file_config).
            print("\n📊 Generating Reports...")
            file_config = get_file_config(system_config.storage)
            output_handler = OutputHandler(
                output_dir=eval_args.output_dir or file_config.output_dir,
                base_filename=file_config.base_filename,
                system_config=system_config,
                file_config=file_config,
            )
            output_handler.generate_reports(results, evaluation_data)

        print("\n🎉 Evaluation Complete!")
        print(f"📊 {len(results)} evaluations completed")
        for fc in file_entries:
            report_dir = Path(eval_args.output_dir or fc.output_dir).resolve()
            print(f"📁 Reports generated in: {report_dir}")
        if not file_entries:
            out_dir = Path(
                eval_args.output_dir
                or get_file_config(system_config.storage).output_dir
            ).resolve()
            print(f"📁 Reports generated in: {out_dir}")

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

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        ConfigurationError,
        DataValidationError,
        StorageError,
    ) as e:
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
        "--tags",
        nargs="+",
        default=None,
        help="Filter by tags (run conversation groups with matching tags)",
    )
    parser.add_argument(
        "--conv-ids",
        nargs="+",
        default=None,
        help="Filter by conversation group IDs (run only specified conversations)",
    )
    parser.add_argument(
        "--cache-warmup",
        action="store_true",
        help="Enable cache warmup mode - rebuild caches without reading existing entries",
    )

    eval_args = parser.parse_args()

    summary = run_evaluation(eval_args)
    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())
