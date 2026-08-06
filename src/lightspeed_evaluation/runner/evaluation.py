"""Lightspeed Evaluation Framework - Main Evaluation Runner."""

import argparse
import logging
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

logger = logging.getLogger(__name__)


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
    run_results: Optional[list] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Print evaluation summary.

    Args:
        totals: Aggregated dict with TOTAL/PASS/FAIL/ERROR/SKIPPED.
        run_results: Run results from orchestrator (agent mode).
        output_dir: Explicit output directory (offline mode).
    """
    print(f"📊 {totals['TOTAL']} evaluations completed")
    if run_results and len(run_results) > 1:
        succeeded = sum(1 for r in run_results if r.success)
        print(
            f"📊 {len(run_results)} runs "
            f"({succeeded} succeeded, {len(run_results) - succeeded} failed)"
        )
    if run_results:
        output_dirs = {
            Path(rr.output_dir).resolve() for rr in run_results if rr.output_dir
        }
        if len(run_results) > 1 and output_dirs:
            common = Path(os.path.commonpath(output_dirs))
            print(f"📁 Reports generated in: {common}")
        else:
            for d in output_dirs:
                print(f"📁 Reports generated in: {d}")
    elif output_dir:
        print(f"📁 Reports generated in: {Path(output_dir).resolve()}")
    print(
        f"✅ Pass: {totals['PASS']}, ❌ Fail: {totals['FAIL']}, "
        f"⚠️ Error: {totals['ERROR']}, ⏭️ Skipped: {totals['SKIPPED']}"
    )
    if totals.get("ERROR", 0) > 0:
        print(f"⚠️ {totals['ERROR']} evaluations had errors - check detailed report")


def _aggregate_totals(run_results: list) -> dict[str, int]:
    """Aggregate status counts from multiple run results."""
    total = passed = failed = error = skipped = 0
    for rr in run_results:
        if rr.summary:
            total += rr.summary.total
            passed += rr.summary.passed
            failed += rr.summary.failed
            error += rr.summary.error
            skipped += rr.summary.skipped
    return {
        "TOTAL": total,
        "PASS": passed,
        "FAIL": failed,
        "ERROR": error,
        "SKIPPED": skipped,
    }


def _copy_flat_output(run_results: list, output_dir: str) -> None:
    """Copy nested output to flat dir for 1x1 backward compatibility.

    Temporary bridge: will be removed once all consumers migrate to
    the nested eval_<timestamp>/agent/run_N/ structure.
    """
    if len(run_results) != 1 or run_results[0].output_dir == output_dir:
        return
    nested = Path(run_results[0].output_dir)
    if not nested.is_dir():
        return
    try:
        flat = Path(output_dir)
        flat.mkdir(parents=True, exist_ok=True)
        shutil.copytree(nested, flat, dirs_exist_ok=True)
    except OSError:
        logger.warning("Failed to copy flat output from %s to %s", nested, output_dir)


def run_evaluation(  # pylint: disable=too-many-locals
    eval_args: argparse.Namespace,
) -> Optional[dict[str, int]]:
    """Run the complete evaluation pipeline.

    Args:
        eval_args: Parsed command line arguments

    Returns:
        dict: Summary statistics with keys TOTAL, PASS, FAIL, ERROR, SKIPPED.
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
            system_config.agents is not None
            and system_config.agents.enabled
            and system_config.agents.default.agent
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
            totals: dict[str, int] = {
                "TOTAL": summary.total,
                "PASS": summary.passed,
                "FAIL": summary.failed,
                "ERROR": summary.error,
                "SKIPPED": summary.skipped,
            }
            print("\n🎉 Evaluation Complete!")
            _print_run_summary(totals, output_dir=out_dir)
            return totals

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
        totals = _aggregate_totals(run_results)
        _copy_flat_output(run_results, output_dir)

        print("\n🎉 Evaluation Complete!")
        _print_run_summary(totals, run_results=run_results)

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
