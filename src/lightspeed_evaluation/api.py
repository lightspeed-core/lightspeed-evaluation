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
"""

from typing import Optional

from lightspeed_evaluation.core.models import (
    EvaluationData,
    EvaluationResult,
    SystemConfig,
    TurnData,
)
from lightspeed_evaluation.core.system import ConfigLoader
from lightspeed_evaluation.pipeline.evaluation import EvaluationPipeline


def evaluate(
    config: SystemConfig,
    data: list[EvaluationData],
    output_dir: Optional[str] = None,
) -> list[EvaluationResult]:
    """Run evaluation on the provided data using the given configuration.

    Creates a fully-initialized pipeline from the ``SystemConfig``, runs
    evaluation on every conversation in *data*, and returns the raw results.
    No reports are generated — file I/O is the caller's responsibility.

    Args:
        config: A pre-built SystemConfig instance.
        data: List of EvaluationData conversations to evaluate.
        output_dir: Optional override for the output directory.

    Returns:
        List of EvaluationResult objects (one per metric per turn/conversation).
    """
    if not data:
        return []

    loader = ConfigLoader.from_config(config)
    pipeline = EvaluationPipeline(loader, output_dir)
    try:
        return pipeline.run_evaluation(data)
    finally:
        pipeline.close()


def evaluate_conversation(
    config: SystemConfig,
    data: EvaluationData,
    output_dir: Optional[str] = None,
) -> list[EvaluationResult]:
    """Evaluate a single conversation group.

    Convenience wrapper around :func:`evaluate` that wraps *data* in a list.

    Args:
        config: A pre-built SystemConfig instance.
        data: A single EvaluationData conversation to evaluate.
        output_dir: Optional override for the output directory.

    Returns:
        List of EvaluationResult objects.
    """
    return evaluate(config, [data], output_dir=output_dir)


def evaluate_turn(
    config: SystemConfig,
    turn: TurnData,
    metrics: Optional[list[str]] = None,
    conversation_group_id: str = "programmatic_eval",
    output_dir: Optional[str] = None,
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

    Returns:
        List of EvaluationResult objects.
    """
    if metrics is not None:
        turn = turn.model_copy(update={"turn_metrics": metrics})

    data = EvaluationData(
        conversation_group_id=conversation_group_id,
        turns=[turn],
    )
    return evaluate(config, [data], output_dir=output_dir)
