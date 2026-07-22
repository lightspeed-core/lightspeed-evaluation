"""NxM evaluation orchestrator.

Runs M agent configurations x N repeats, saving results independently
per run. Each run calls the pipeline directly with a cloned config
targeting one agent.
"""

import copy
import logging
import multiprocessing
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from typing import Any, Optional

from lightspeed_evaluation.core.models import EvaluationData, SystemConfig
from lightspeed_evaluation.core.models.data import DatasetMetadata
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.core.system.exceptions import (
    ConfigurationError,
    DataValidationError,
)
from lightspeed_evaluation.pipeline.behavioral.models import RunContext, RunResult
from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline

logger = logging.getLogger(__name__)


def run(
    system_config: SystemConfig,
    evaluation_data: list[EvaluationData],
    output_base: str,
    *,
    original_data_path: Optional[str] = None,
    dataset_metadata_dict: Optional[dict[str, Any]] = None,
) -> list[RunResult]:
    """Run NxM evaluation matrix.

    Args:
        system_config: System configuration with agents, repeat, parallel.
        evaluation_data: Conversation groups to evaluate.
        output_base: Base output directory.
        original_data_path: Path to original eval data file.
        dataset_metadata_dict: Serialized dataset metadata to pass through.

    Returns:
        List of RunResult with metadata per run.
    """
    agents_config = system_config.agents
    if agents_config is None or not agents_config.default.agent:
        logger.warning("No agents configured, nothing to orchestrate")
        return []

    default_agents = agents_config.default.agent
    repeat = agents_config.default.repeat
    parallel = agents_config.default.parallel
    all_agents = _build_agent_set(default_agents, evaluation_data)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    run_matrix = [(agent, i) for agent in all_agents for i in range(1, repeat + 1)]
    if not run_matrix:
        logger.warning("No runs to execute")
        return []

    logger.info(
        "Starting NxM evaluation: %d agents x %d repeats = %d runs",
        len(all_agents),
        repeat,
        len(run_matrix),
    )

    contexts = _build_run_contexts(
        system_config,
        evaluation_data,
        {
            "default_agents": default_agents,
            "run_matrix": run_matrix,
            "output_base": output_base,
            "timestamp": timestamp,
            "original_data_path": original_data_path,
            "dataset_metadata_dict": dataset_metadata_dict,
        },
    )

    if parallel and len(contexts) > 1:
        _warn_resource_usage(len(contexts), system_config)
        results = _run_parallel(contexts)
    else:
        results = _run_sequential(contexts)

    succeeded = sum(1 for r in results if r.success)
    logger.info(
        "NxM evaluation complete: %d/%d succeeded, %d failed",
        succeeded,
        len(results),
        len(results) - succeeded,
    )
    return results


def _build_agent_set(
    default_agents: list[str],
    evaluation_data: list[EvaluationData],
) -> list[str]:
    """Build deduplicated agent set from config defaults + eval data overrides."""
    agents: dict[str, None] = {}
    for agent in default_agents:
        agents[agent] = None
    for conv in evaluation_data:
        if conv.agent:
            for agent in conv.agent:
                agents[agent] = None
    return list(agents.keys())


def _build_run_contexts(
    system_config: SystemConfig,
    evaluation_data: list[EvaluationData],
    run_params: dict[str, Any],
) -> list[RunContext]:
    """Build serializable RunContext for each run in the matrix.

    Args:
        system_config: System configuration to serialize.
        evaluation_data: Conversation groups to serialize.
        run_params: Dict with keys: default_agents, run_matrix,
            output_base, timestamp, original_data_path.
    """
    config_dict = system_config.model_dump()
    eval_data_dicts = [conv.model_dump() for conv in evaluation_data]
    output_base = run_params["output_base"]
    timestamp = run_params["timestamp"]

    return [
        RunContext(
            config_dict=config_dict,
            eval_data_dicts=eval_data_dicts,
            default_agents=run_params["default_agents"],
            agent_name=agent,
            run_index=idx,
            run_output_dir=os.path.join(
                output_base, f"eval_{timestamp}", agent, f"run_{idx}"
            ),
            extra={
                "original_data_path": run_params.get("original_data_path"),
                "dataset_metadata_dict": run_params.get("dataset_metadata_dict"),
            },
        )
        for agent, idx in run_params["run_matrix"]
    ]


def _filter_conversations(
    evaluation_data: list[EvaluationData],
    agent_name: str,
    default_agents: list[str],
) -> list[EvaluationData]:
    """Filter conversations that should participate in this agent's run."""
    return [
        conv for conv in evaluation_data if agent_name in (conv.agent or default_agents)
    ]


def _clone_config_for_run(
    config_dict: dict[str, Any],
    agent_name: str,
    disable_cache: bool,
) -> dict[str, Any]:
    """Clone config dict and set the target agent as the sole default."""
    cloned = copy.deepcopy(config_dict)
    cloned["agents"]["default"]["agent"] = [agent_name]

    agent_defs = cloned.get("agents", {}).get("agents", {})
    if disable_cache and agent_name in agent_defs:
        agent_def = agent_defs[agent_name]
        if isinstance(agent_def, dict):
            agent_def["cache_enabled"] = False

    return cloned


def _run_sequential(contexts: list[RunContext]) -> list[RunResult]:
    """Execute runs sequentially."""
    results: list[RunResult] = []
    total = len(contexts)

    for idx, ctx in enumerate(contexts, 1):
        result = _run_single(ctx)
        results.append(result)
        status = "OK" if result.success else "FAILED"
        logger.info(
            "[%d/%d] %s/run_%d %s", idx, total, ctx.agent_name, ctx.run_index, status
        )

    return results


def _run_parallel(contexts: list[RunContext]) -> list[RunResult]:
    """Execute runs in parallel using ProcessPoolExecutor."""
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    results: list[RunResult] = []
    completed = 0
    total = len(contexts)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_ctx = {executor.submit(_run_single, ctx): ctx for ctx in contexts}

        for future in as_completed(future_to_ctx):
            ctx = future_to_ctx[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)
                status = "OK" if result.success else "FAILED"
            except (ConfigurationError, DataValidationError):
                raise
            except Exception:  # pylint: disable=broad-exception-caught
                results.append(
                    RunResult(
                        agent_name=ctx.agent_name,
                        run_index=ctx.run_index,
                        output_dir="",
                        success=False,
                        error=traceback.format_exc(),
                    )
                )
                status = "ERROR"

            logger.info(
                "[%d/%d] %s/run_%d %s",
                completed,
                total,
                ctx.agent_name,
                ctx.run_index,
                status,
            )

    return results


def _pin_conversations_to_agent(
    conversations: list[EvaluationData],
    agent_name: str,
) -> list[EvaluationData]:
    """Pin each conversation's agent list to the current run's agent.

    Conversations with multi-agent lists (e.g. agent: [model_a, model_c])
    are narrowed to just the current agent so the pipeline resolves the
    correct driver.
    """
    pinned: list[EvaluationData] = []
    for conv in conversations:
        if conv.agent and len(conv.agent) > 1:
            conv = conv.model_copy(update={"agent": [agent_name]})
        pinned.append(conv)
    return pinned


def _run_single(ctx: RunContext) -> RunResult:
    """Execute one evaluation run. Module-level for ProcessPoolExecutor pickling."""
    os.makedirs(ctx.run_output_dir, exist_ok=True)

    try:
        cloned_dict = _clone_config_for_run(
            ctx.config_dict, ctx.agent_name, disable_cache=ctx.run_index > 1
        )
        config = SystemConfig.model_validate(cloned_dict)

        all_conversations = [
            EvaluationData.model_validate(d) for d in ctx.eval_data_dicts
        ]
        filtered = _filter_conversations(
            all_conversations, ctx.agent_name, ctx.default_agents
        )

        if not filtered:
            return RunResult(
                agent_name=ctx.agent_name,
                run_index=ctx.run_index,
                output_dir=ctx.run_output_dir,
                success=True,
                summary=_empty_summary(),
            )

        pinned = _pin_conversations_to_agent(filtered, ctx.agent_name)

        extra = ctx.extra or {}
        dataset_metadata = None
        metadata_dict = extra.get("dataset_metadata_dict")
        if metadata_dict:
            dataset_metadata = DatasetMetadata.model_validate(metadata_dict)

        loader = ConfigLoader.from_config(config)
        pipeline = EvaluationPipeline(loader, ctx.run_output_dir)
        try:
            eval_results = pipeline.run_evaluation(
                pinned,
                original_data_path=extra.get("original_data_path"),
                dataset_metadata=dataset_metadata,
            )
        finally:
            pipeline.close()

        return RunResult(
            agent_name=ctx.agent_name,
            run_index=ctx.run_index,
            output_dir=ctx.run_output_dir,
            success=True,
            summary=_make_summary(eval_results),
        )

    except (ConfigurationError, DataValidationError):
        raise
    except Exception:  # pylint: disable=broad-exception-caught
        logger.error(
            "Run %s/run_%d failed: %s",
            ctx.agent_name,
            ctx.run_index,
            traceback.format_exc(),
        )
        return RunResult(
            agent_name=ctx.agent_name,
            run_index=ctx.run_index,
            output_dir=ctx.run_output_dir,
            success=False,
            error=traceback.format_exc(),
        )


def _make_summary(results: list) -> dict[str, int]:
    """Build summary dict from evaluation results."""
    return {
        "TOTAL": len(results),
        "PASS": sum(1 for r in results if r.result == "PASS"),
        "FAIL": sum(1 for r in results if r.result == "FAIL"),
        "ERROR": sum(1 for r in results if r.result == "ERROR"),
        "SKIPPED": sum(1 for r in results if r.result == "SKIPPED"),
    }


def _empty_summary() -> dict[str, int]:
    """Return empty summary for runs with no matching conversations."""
    return {"TOTAL": 0, "PASS": 0, "FAIL": 0, "ERROR": 0, "SKIPPED": 0}


def _warn_resource_usage(num_runs: int, config: SystemConfig) -> None:
    """Log warning if parallel runs may overwhelm the system."""
    max_threads = config.core.max_threads or 1
    cpu_count = multiprocessing.cpu_count()
    total_threads = num_runs * max_threads

    if total_threads > cpu_count * 2:
        logger.warning(
            "High resource usage: %d parallel runs x %d threads = "
            "%d total threads on %d CPUs. Consider reducing "
            "core.max_threads or the number of agents/repeats.",
            num_runs,
            max_threads,
            total_threads,
            cpu_count,
        )
