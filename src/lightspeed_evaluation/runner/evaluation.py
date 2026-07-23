"""Lightspeed Evaluation Framework - Main Evaluation Runner."""

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

from lightspeed_evaluation.core.models import (
    LLMPoolConfig,
    SystemConfig,
)

# Import only lightweight modules at top level
from lightspeed_evaluation.core.storage import get_file_config
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
    pool_cache_path = None
    if (
        isinstance(pool, LLMPoolConfig)
        and pool.defaults.cache_enabled
        and pool.defaults.cache_dir
    ):
        pool_cache_path = Path(pool.defaults.cache_dir).resolve()
        cache_dirs.append(("LLM Judge (pool)", pool.defaults.cache_dir))
    if system_config.llm.cache_enabled and system_config.llm.cache_dir:
        llm_cache_path = Path(system_config.llm.cache_dir).resolve()
        # Skip if same as pool cache to avoid duplicate clearing
        if llm_cache_path != pool_cache_path:
            cache_dirs.append(("LLM Judge", system_config.llm.cache_dir))
    # We clear the api cache even if the Lightspeed core api is disabled
    if system_config.api.cache_enabled and system_config.api.cache_dir:
        cache_dirs.append(("API", system_config.api.cache_dir))

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


def _print_run_summary(
    totals: dict[str, int],
    run_results: list,
) -> None:
    """Print evaluation summary and token usage from orchestrator results."""
    print(f"📊 {totals['TOTAL']} evaluations completed")
    if len(run_results) > 1:
        succeeded = sum(1 for r in run_results if r.success)
        print(
            f"📊 {len(run_results)} runs "
            f"({succeeded} succeeded, {len(run_results) - succeeded} failed)"
        )
    output_dirs = {Path(rr.output_dir).resolve() for rr in run_results if rr.output_dir}
    if len(run_results) > 1 and output_dirs:
        common = Path(os.path.commonpath(output_dirs))
        print(f"📁 Reports generated in: {common}")
    else:
        for d in output_dirs:
            print(f"📁 Reports generated in: {d}")
    print(
        f"✅ Pass: {totals['PASS']}, ❌ Fail: {totals['FAIL']}, "
        f"⚠️ Error: {totals['ERROR']}, ⏭️ Skipped: {totals['SKIPPED']}"
    )
    if totals.get("ERROR", 0) > 0:
        print(f"⚠️ {totals['ERROR']} evaluations had errors - check detailed report")

    judge_in = totals.get("judge_llm_input_tokens", 0)
    judge_out = totals.get("judge_llm_output_tokens", 0)
    embed = totals.get("embedding_tokens", 0)
    api_in = totals.get("api_input_tokens", 0)
    api_out = totals.get("api_output_tokens", 0)

    print("\n📊 Token Usage Summary:")
    print(
        f"Judge LLM: {judge_in + judge_out:,} tokens "
        f"(Input: {judge_in:,}, Output: {judge_out:,})"
    )
    print(f"Embeddings: {embed:,} tokens")
    if api_in + api_out > 0:
        print(
            f"API Calls: {api_in + api_out:,} tokens "
            f"(Input: {api_in:,}, Output: {api_out:,})"
        )
    total_tokens = judge_in + judge_out + embed + api_in + api_out
    if total_tokens > 0:
        print(f"Total: {total_tokens:,} tokens")


def run_evaluation(  # pylint: disable=too-many-locals
    eval_args: argparse.Namespace,
) -> Optional[dict[str, int]]:
    """Run the complete evaluation pipeline.

    Args:
        eval_args: Parsed command line arguments

    Returns:
        dict: Summary statistics with keys TOTAL, PASS, FAIL, ERROR, SKIPPED
    """
    print("🚀 Lightspeed Evaluation Framework")
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
        from lightspeed_evaluation.core.output.statistics import compute_overall_stats
        from lightspeed_evaluation.core.storage import FileBackendConfig
        from lightspeed_evaluation.core.system import DataValidator
        from lightspeed_evaluation.pipeline.behavioral.orchestrator import (
            run as orchestrator_run,
        )

        # pylint: enable=import-outside-toplevel
        print("✅ Configuration loaded & Setup is done !")

        # Load, filter, and validate evaluation data
        data_validator = DataValidator(
            api_enabled=system_config.agents is not None
            and system_config.agents.enabled,
            fail_on_invalid_data=system_config.core.fail_on_invalid_data,
            system_config=system_config,
        )
        evaluation_data = data_validator.load_evaluation_data(
            eval_args.eval_data,
            tags=eval_args.tags,
            conv_ids=eval_args.conv_ids,
            metrics=eval_args.metrics,
        )
        dataset_metadata = data_validator.dataset_metadata

        print(
            f"✅ System config: {system_config.llm.provider}/{system_config.llm.model}"
        )

        # Handle case where no conversations match the filter
        if len(evaluation_data) == 0:
            print("\n⚠️ No conversation groups matched the filter criteria")
            print("   Nothing to evaluate - returning empty results")
            return {"TOTAL": 0, "PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}

        # Run evaluation
        print("\n🔄 Running Evaluation...")
        has_agents = (
            system_config.agents is not None and system_config.agents.default.agent
        )

        if not has_agents:
            # Offline mode: run pipeline directly (no agents to orchestrate)
            results = evaluate(
                system_config,
                evaluation_data,
                output_dir=eval_args.output_dir,
                original_data_path=eval_args.eval_data,
                dataset_metadata=dataset_metadata,
            )
            file_entries = [
                c for c in system_config.storage if isinstance(c, FileBackendConfig)
            ]
            if not file_entries:
                file_config = get_file_config(system_config.storage)
                handler = OutputHandler(
                    output_dir=eval_args.output_dir or file_config.output_dir,
                    base_filename=file_config.base_filename,
                    system_config=system_config,
                    file_config=file_config,
                )
                handler.generate_reports(results, evaluation_data)
            summary = compute_overall_stats(results)
            out_dir = (
                eval_args.output_dir
                or get_file_config(system_config.storage).output_dir
            )
            print(f"\n🎉 Evaluation Complete!\n📊 {len(results)} evaluations completed")
            print(f"📁 Reports generated in: {Path(out_dir).resolve()}")
            print(
                f"✅ Pass: {summary.passed}, ❌ Fail: {summary.failed}, "
                f"⚠️ Error: {summary.error}, ⏭️ Skipped: {summary.skipped}"
            )
            print(
                f"\n📊 Token Usage Summary:\n"
                f"Judge LLM: {summary.total_judge_llm_tokens:,} tokens "
                f"(Input: {summary.total_judge_llm_input_tokens:,}, "
                f"Output: {summary.total_judge_llm_output_tokens:,})\n"
                f"Embeddings: {summary.total_embedding_tokens:,} tokens"
            )
            return {
                "TOTAL": summary.total,
                "PASS": summary.passed,
                "FAIL": summary.failed,
                "ERROR": summary.error,
                "SKIPPED": summary.skipped,
            }

        # Agent mode: run via orchestrator
        output_dir = (
            eval_args.output_dir or get_file_config(system_config.storage).output_dir
        )
        run_results = orchestrator_run(
            system_config,
            evaluation_data,
            output_dir,
            original_data_path=eval_args.eval_data,
            dataset_metadata_dict=(
                dataset_metadata.model_dump() if dataset_metadata else None
            ),
        )
        totals: dict[str, int] = {
            "TOTAL": 0,
            "PASS": 0,
            "FAIL": 0,
            "ERROR": 0,
            "SKIPPED": 0,
        }
        for rr in run_results:
            if rr.summary:
                for key, val in rr.summary.items():
                    totals[key] = totals.get(key, 0) + val

        print("\n🎉 Evaluation Complete!")
        _print_run_summary(totals, run_results)

        return totals

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


def create_eval_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Lightspeed Evaluation Framework / Tool",
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
        "--metrics",
        nargs="+",
        default=None,
        help="Filter to only run specified metrics (e.g. custom:answer_correctness)",
    )
    parser.add_argument(
        "--cache-warmup",
        action="store_true",
        help="Enable cache warmup mode - rebuild caches without reading existing entries",
    )
    return parser


def main() -> int:
    """Command line interface."""
    eval_args = create_eval_parser().parse_args()

    summary = run_evaluation(eval_args)
    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main())
