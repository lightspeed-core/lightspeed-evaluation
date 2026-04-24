"""Programmatic API for the LightSpeed Evaluation Framework.

Provides clean public functions for using the framework as a Python library,
without requiring YAML files or CLI argument parsing.

Example usage::

    from lightspeed_evaluation import evaluate, SystemConfig, EvaluationData, TurnData

    config = SystemConfig(llm=LLMConfig(provider="openai", model="gpt-4o-mini"))
    data = EvaluationData(
        conversation_group_id="my_eval",
        turns=[TurnData(turn_id="t1", query="What is OCP?", response="...")],
    )
    results = evaluate(config, [data])

For structured results with computed statistics::

    from lightspeed_evaluation import evaluate_with_summary

    summary = evaluate_with_summary(config, [data])
    print(summary.overall.pass_rate)
    print(summary.by_metric)
"""

from collections.abc import Callable
from typing import Optional

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    EvaluationRunContext,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.models.summary import EvaluationSummary
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline


def evaluate(  # pylint: disable=too-many-arguments
    config: SystemConfig,
    data: list[EvaluationData],
    output_dir: Optional[str] = None,
    *,
    evaluation_data_path: Optional[str] = None,
    on_complete: Optional[
        Callable[[list[EvaluationResult], EvaluationRunContext], None]
    ] = None,
) -> list[EvaluationResult]:
    """Run evaluation on the provided data using the given configuration.

    Creates a fully-initialized pipeline from the ``SystemConfig``, runs
    evaluation on every conversation in *data*, and returns the raw results.
    No reports are generated -- file I/O is the caller's responsibility.

    Args:
        config: A pre-built SystemConfig instance.
        data: List of EvaluationData conversations to evaluate.
        output_dir: Optional override for the output directory.
        evaluation_data_path: Optional path to the evaluation data file, used
            for run naming and in :class:`EvaluationRunContext` (e.g. Langfuse).
        on_complete: Optional callback after a successful run; receives results
            and an :class:`EvaluationRunContext`. See
            :mod:`lightspeed_evaluation.integrations.langfuse_reporter` for
            a Langfuse helper. Failures in the callback do not fail the run.

    Returns:
        List of EvaluationResult objects (one per metric per turn/conversation).
    """
    if not data:
        return []

    loader = ConfigLoader.from_config(config)
    pipeline = EvaluationPipeline(loader, output_dir)
    try:
        return pipeline.run_evaluation(
            data,
            original_data_path=evaluation_data_path,
            on_complete=on_complete,
        )
    finally:
        pipeline.close()


def evaluate_with_summary(  # pylint: disable=too-many-arguments
    config: SystemConfig,
    data: list[EvaluationData],
    output_dir: Optional[str] = None,
    compute_confidence_intervals: bool = False,
    *,
    evaluation_data_path: Optional[str] = None,
    on_complete: Optional[
        Callable[[list[EvaluationResult], EvaluationRunContext], None]
    ] = None,
) -> EvaluationSummary:
    """Run evaluation and return structured results with computed statistics.

    Like :func:`evaluate`, but wraps the raw results in an
    :class:`EvaluationSummary` that includes overall, per-metric,
    per-conversation, and per-tag statistics.

    Args:
        config: A pre-built SystemConfig instance.
        data: List of EvaluationData conversations to evaluate.
        output_dir: Optional override for the output directory.
        compute_confidence_intervals: Whether to compute bootstrap confidence
            intervals. Default False.
        evaluation_data_path: Same as for :func:`evaluate`.
        on_complete: Same as for :func:`evaluate`.

    Returns:
        EvaluationSummary with results and computed statistics.
    """
    results = evaluate(
        config,
        data,
        output_dir=output_dir,
        evaluation_data_path=evaluation_data_path,
        on_complete=on_complete,
    )
    return EvaluationSummary.from_results(
        results,
        evaluation_data=data if data else None,
        compute_confidence_intervals=compute_confidence_intervals,
    )


def evaluate_conversation(  # pylint: disable=too-many-arguments
    config: SystemConfig,
    data: EvaluationData,
    output_dir: Optional[str] = None,
    *,
    evaluation_data_path: Optional[str] = None,
    on_complete: Optional[
        Callable[[list[EvaluationResult], EvaluationRunContext], None]
    ] = None,
) -> list[EvaluationResult]:
    """Evaluate a single conversation group.

    Convenience wrapper around :func:`evaluate` that wraps *data* in a list.

    Args:
        config: A pre-built SystemConfig instance.
        data: A single EvaluationData conversation to evaluate.
        output_dir: Optional override for the output directory.
        evaluation_data_path: Same as for :func:`evaluate`.
        on_complete: Same as for :func:`evaluate`.

    Returns:
        List of EvaluationResult objects.
    """
    return evaluate(
        config,
        [data],
        output_dir=output_dir,
        evaluation_data_path=evaluation_data_path,
        on_complete=on_complete,
    )


def evaluate_conversation_with_summary(  # pylint: disable=too-many-arguments
    config: SystemConfig,
    data: EvaluationData,
    output_dir: Optional[str] = None,
    compute_confidence_intervals: bool = False,
    *,
    evaluation_data_path: Optional[str] = None,
    on_complete: Optional[
        Callable[[list[EvaluationResult], EvaluationRunContext], None]
    ] = None,
) -> EvaluationSummary:
    """Evaluate a single conversation and return structured results.

    Convenience wrapper around :func:`evaluate_with_summary` for one conversation.

    Args:
        config: A pre-built SystemConfig instance.
        data: A single EvaluationData conversation to evaluate.
        output_dir: Optional override for the output directory.
        compute_confidence_intervals: Whether to compute bootstrap confidence
            intervals. Default False.
        evaluation_data_path: Same as for :func:`evaluate`.
        on_complete: Same as for :func:`evaluate`.

    Returns:
        EvaluationSummary with results and computed statistics.
    """
    return evaluate_with_summary(
        config,
        [data],
        output_dir=output_dir,
        compute_confidence_intervals=compute_confidence_intervals,
        evaluation_data_path=evaluation_data_path,
        on_complete=on_complete,
    )


def evaluate_turn(  # pylint: disable=too-many-arguments
    config: SystemConfig,
    turn: TurnData,
    metrics: Optional[list[str]] = None,
    conversation_group_id: str = "programmatic_eval",
    output_dir: Optional[str] = None,
    *,
    evaluation_data_path: Optional[str] = None,
    on_complete: Optional[
        Callable[[list[EvaluationResult], EvaluationRunContext], None]
    ] = None,
) -> list[EvaluationResult]:
    """Evaluate a single turn.

    Wraps the turn in an :class:`EvaluationData` instance and delegates to
    :func:`evaluate`. If *metrics* is provided, a copy of the turn is created
    with updated ``turn_metrics``.

    Args:
        config: A pre-built SystemConfig instance.
        turn: The TurnData to evaluate.
        metrics: Optional list of metric identifiers to override turn_metrics.
        conversation_group_id: Conversation group ID for the wrapper.
        output_dir: Optional override for the output directory.
        evaluation_data_path: Same as for :func:`evaluate`.
        on_complete: Same as for :func:`evaluate`.

    Returns:
        List of EvaluationResult objects.
    """
    if metrics is not None:
        turn = TurnData.model_validate({**turn.model_dump(), "turn_metrics": metrics})

    data = EvaluationData(
        conversation_group_id=conversation_group_id,
        turns=[turn],
    )
    return evaluate(
        config,
        [data],
        output_dir=output_dir,
        evaluation_data_path=evaluation_data_path,
        on_complete=on_complete,
    )


def evaluate_turn_with_summary(  # pylint: disable=too-many-arguments
    config: SystemConfig,
    turn: TurnData,
    metrics: Optional[list[str]] = None,
    conversation_group_id: str = "programmatic_eval",
    output_dir: Optional[str] = None,
    *,
    evaluation_data_path: Optional[str] = None,
    on_complete: Optional[
        Callable[[list[EvaluationResult], EvaluationRunContext], None]
    ] = None,
) -> EvaluationSummary:
    """Evaluate a single turn and return structured results.

    Like :func:`evaluate_turn`, but returns an :class:`EvaluationSummary`.

    Args:
        config: A pre-built SystemConfig instance.
        turn: The TurnData to evaluate.
        metrics: Optional list of metric identifiers to override turn_metrics.
        conversation_group_id: Conversation group ID for the wrapper.
        output_dir: Optional override for the output directory.
        evaluation_data_path: Same as for :func:`evaluate`.
        on_complete: Same as for :func:`evaluate`.

    Returns:
        EvaluationSummary with results and computed statistics.
    """
    if metrics is not None:
        turn = TurnData.model_validate({**turn.model_dump(), "turn_metrics": metrics})

    data = EvaluationData(
        conversation_group_id=conversation_group_id,
        turns=[turn],
    )
    return evaluate_with_summary(
        config,
        [data],
        output_dir=output_dir,
        compute_confidence_intervals=False,
        evaluation_data_path=evaluation_data_path,
        on_complete=on_complete,
    )
