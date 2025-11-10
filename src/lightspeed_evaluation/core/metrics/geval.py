"""GEval metrics handler using LLM Manager.

This module provides integration with DeepEval's GEval for configurable custom evaluation criteria.
GEval allows runtime-defined evaluation metrics through YAML configuration.
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from lightspeed_evaluation.core.llm.deepeval import DeepEvalLLMManager

logger = logging.getLogger(__name__)


class GEvalHandler:  # pylint: disable=R0903
    """Handler for configurable GEval metrics.

    This class integrates with the lightspeed-evaluation framework
    to provide GEval evaluation with criteria defined either in:
    1. A centralized metric registry (config/registry/geval_metrics.yaml)
    2. Runtime YAML configuration (turn_metrics_metadata)

    Priority: Runtime metadata overrides registry definitions.
    """

    # Class-level registry cache (shared across instances)
    _registry: dict[str, Any] | None = None
    _registry_path: Path | None = None

    def __init__(
        self,
        deepeval_llm_manager: DeepEvalLLMManager,
        registry_path: str | None = None,
    ) -> None:
        """Initialize GEval handler.

        Args:
            deepeval_llm_manager: Shared DeepEvalLLMManager instance
            registry_path: Optional path to metric registry YAML.
                          If not provided, looks for config/registry/geval_metrics.yaml
                          relative to project root.
        """
        self.deepeval_llm_manager = deepeval_llm_manager
        self._load_registry(registry_path)

    def _load_registry(self, registry_path: str | None = None) -> None:
        """Load the GEval metric registry from a YAML configuration file.

        This method initializes the class-level `_registry`.
        It supports both user-specified and auto-discovered paths, searching common
        locations relative to the current working directory and the package root.

        If no valid registry file is found, it logs a warning and initializes an
        empty registry (meaning GEval will rely solely on runtime metadata).

        Args:
            registry_path (str | None): Optional explicit path to a registry YAML file.

        Behavior:
            - If the registry has already been loaded, the function returns immediately.
            - If `registry_path` is provided, it is used directly.
            - Otherwise, common fallback paths are checked for existence.
            - If a registry is found, it is parsed with `yaml.safe_load`.
            - Any exceptions during file access or parsing are logged, and an empty
            registry is used as a fallback.
        """
        # Only load once per class
        if GEvalHandler._registry is not None:
            return

        # Determine registry path
        possible_paths = []
        if registry_path:
            path = Path(registry_path)
            possible_paths = [path]
        else:
            # Look for config/registry/geval_metrics.yaml relative to project root
            # Try multiple locations
            possible_paths = [
                Path.cwd() / "config" / "registry" / "geval_metrics.yaml",
                Path(__file__).parent.parent.parent.parent
                / "config"
                / "registry"
                / "geval_metrics.yaml",
            ]
            path = None
            for p in possible_paths:
                if p.exists():
                    path = p
                    break
        # Handle missing or invalid registry
        if path is None or not path.exists():
            logger.warning(
                "GEval metric registry not found at expected locations. "
                "Tried: %s. Will fall back to runtime metadata only.",
                [str(p) for p in possible_paths],
            )
            GEvalHandler._registry = {}
            return

        # Load registry file
        try:
            with open(path, encoding="utf-8") as f:
                GEvalHandler._registry = (
                    yaml.safe_load(f) or {}
                )  # Default to empty dict if file is empty
                GEvalHandler._registry_path = path
                num_metrics = (
                    len(GEvalHandler._registry) if GEvalHandler._registry else 0
                )
                logger.info("Loaded %d GEval metrics from %s", num_metrics, path)
        except Exception as e:  # pylint: disable=W0718
            logger.error("Failed to load GEval registry from %s: %s", path, e)
            GEvalHandler._registry = {}

    def evaluate(  # pylint: disable=R0913,R0917
        self,
        metric_name: str,
        conv_data: Any,
        _turn_idx: int | None,
        turn_data: Any | None,
        is_conversation: bool,
    ) -> tuple[float | None, str]:
        """Evaluate using GEval with runtime configuration.

        This method is the central entry point for running GEval evaluations.
        It retrieves the appropriate metric configuration (from registry or runtime
        metadata), extracts evaluation parameters, and delegates the actual scoring
        to either conversation-level or turn-level evaluators.

         Args:
            metric_name (str):
                The name of the metric to evaluate (e.g., "technical_accuracy").
            conv_data (Any):
                The conversation data object containing context, messages, and
                associated metadata.
            turn_idx (int | None):
                The index of the current turn in the conversation.
                (Currently unused but kept for interface compatibility.)
            turn_data (Any | None):
                The turn-level data object, required when evaluating turn-level metrics.
            is_conversation (bool):
                Indicates whether the evaluation should run on the entire
                conversation (`True`) or on an individual turn (`False`).

        Returns:
        tuple[float | None, str]:
            A tuple containing:
              - **score** (float | None): The computed metric score, or None if evaluation failed.
              - **reason** (str): A descriptive reason or error message.

        Behavior:
        1. Fetch GEval configuration from metadata using `_get_geval_config()`.
        2. Validate that required fields (e.g., "criteria") are present.
        3. Extract key parameters such as criteria, evaluation steps, and threshold.
        4. Delegate to `_evaluate_conversation()` or `_evaluate_turn()` depending
           on the `is_conversation` flag.
        """
        # Extract GEval configuration from metadata
        # May come from runtime metadata or a preloaded registry
        geval_config = self._get_geval_config(
            metric_name, conv_data, turn_data, is_conversation
        )

        # If no configuration is available, return early with an informative message.
        if not geval_config:
            return None, f"GEval configuration not found for metric '{metric_name}'"

        # Extract configuration parameters
        criteria = geval_config.get("criteria")
        evaluation_params = geval_config.get("evaluation_params", [])
        evaluation_steps = geval_config.get("evaluation_steps")
        threshold = geval_config.get("threshold", 0.5)

        # The criteria field defines what the model is being judged on.
        # Without it, we cannot perform evaluation. Evaluation steps can be generated
        if not criteria:
            return None, "GEval requires 'criteria' in configuration"

        # Perform evaluation based on level (turn or conversation)
        if is_conversation:
            return self._evaluate_conversation(
                conv_data, criteria, evaluation_params, evaluation_steps, threshold
            )
        return self._evaluate_turn(
            turn_data, criteria, evaluation_params, evaluation_steps, threshold
        )

    def _convert_evaluation_params(
        self, params: list[str]
    ) -> list[LLMTestCaseParams] | None:
        """Convert a list of string parameter names into `LLMTestCaseParams` enum values.

        This helper ensures that the evaluation parameters passed into GEval are properly
        typed as `LLMTestCaseParams` (used by DeepEval's test-case schema). If any
        parameter is not a valid enum member, the function treats the entire parameter
        list as "custom" and returns `None`. This allows GEval to automatically infer
        the required fields at runtime rather than forcing strict schema compliance.

        Args:
            params (list[str]):
                A list of string identifiers (e.g., ["input", "actual_output"]).
                These typically come from a YAML or runtime configuration and
                may not always match enum names exactly.

        Returns:
            List of LLMTestCaseParams enum values, or None if params are custom strings
        """
        # Return early if no parameters were supplied
        if not params:
            return None

        # Try to convert strings to enum values
        converted: list[LLMTestCaseParams] = []

        # Attempt to convert each string into a valid enum value
        for param in params:
            try:
                # Try to match as enum value (e.g., "INPUT", "ACTUAL_OUTPUT")
                enum_value = LLMTestCaseParams[param.upper().replace(" ", "_")]
                converted.append(enum_value)
            except (KeyError, AttributeError):
                # Not a valid enum - these are custom params, skip them
                logger.debug(
                    "Skipping custom evaluation_param '%s' - "
                    "not a valid LLMTestCaseParams enum. "
                    "GEval will auto-detect required fields.",
                    param,
                )
                return None

        # Return the successfully converted list, or None if it ended up empty
        return converted if converted else None

    def _evaluate_turn(  # pylint: disable=R0913,R0917
        self,
        turn_data: Any,
        criteria: str,
        evaluation_params: list[str],
        evaluation_steps: list[str] | None,
        threshold: float,
    ) -> tuple[float | None, str]:
        """Evaluate a single turn using GEval.

            Args:
            turn_data (Any):
                The turn-level data object containing fields like query, response,
                expected_response, and context.
            criteria (str):
                Natural-language description of what the evaluation should judge.
                Example: "Assess factual correctness and command validity."
            evaluation_params (list[str]):
                A list of string parameters defining which fields to include
                (e.g., ["input", "actual_output"]).
            evaluation_steps (list[str] | None):
                Optional step-by-step evaluation guidance for the model.
            threshold (float):
                Minimum score threshold that determines pass/fail behavior.

        Returns:
            tuple[float | None, str]:
                A tuple of (score, reason). If evaluation fails, score will be None
                and the reason will contain an error message.
        """
        # Validate that we actually have turn data
        if not turn_data:
            return None, "Turn data required for turn-level GEval"

        # Convert evaluation_params to enum values if valid, otherwise use defaults
        converted_params = self._convert_evaluation_params(evaluation_params)
        if not converted_params:
            # If no valid params, use sensible defaults for turn evaluation
            converted_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ]

        # Create GEval metric with runtime configuration
        metric_kwargs: dict[str, Any] = {
            "name": "GEval Turn Metric",
            "criteria": criteria,
            "evaluation_params": converted_params,
            "model": self.deepeval_llm_manager.get_llm(),
            "threshold": threshold,
            "top_logprobs": 5,
        }

        # Add evaluation steps if provided
        if evaluation_steps:
            metric_kwargs["evaluation_steps"] = evaluation_steps

        # Instantiate the GEval metric object
        metric = GEval(**metric_kwargs)

        # Prepare test case arguments, only including non-None optional fields
        test_case_kwargs = {
            "input": turn_data.query,
            "actual_output": turn_data.response or "",
        }

        # Add optional fields only if they have values
        if turn_data.expected_response:
            test_case_kwargs["expected_output"] = turn_data.expected_response

        if turn_data.contexts:
            # Normalize contexts: handle both dict and string formats
            normalized_contexts = [
                ctx.get("content", str(ctx)) if isinstance(ctx, dict) else str(ctx)
                for ctx in turn_data.contexts
            ]
            test_case_kwargs["context"] = normalized_contexts

        # Create test case for a single turn
        test_case = LLMTestCase(**test_case_kwargs)

        # Evaluate
        try:
            metric.measure(test_case)
            score = metric.score if metric.score is not None else 0.0
            reason = (
                str(metric.reason)
                if hasattr(metric, "reason") and metric.reason
                else "No reason provided"
            )
            return score, reason
        except Exception as e:  # pylint: disable=W0718
            logger.error(
                "GEval turn-level evaluation failed: %s: %s", type(e).__name__, str(e)
            )
            logger.debug(
                "Test case input: %s...",
                test_case.input[:100] if test_case.input else "None",
            )
            logger.debug(
                "Test case output: %s...",
                test_case.actual_output[:100] if test_case.actual_output else "None",
            )
            return None, f"GEval evaluation error: {str(e)}"

    def _evaluate_conversation(  # pylint: disable=R0913,R0917,R0914
        self,
        conv_data: Any,
        criteria: str,
        evaluation_params: list[str],
        evaluation_steps: list[str] | None,
        threshold: float,
    ) -> tuple[float | None, str]:
        """Evaluate a conversation using GEval.

        This method aggregates all conversation turns into a single LLMTestCase
        and evaluates the conversation against the provided criteria.

        Args:
            conv_data (Any):
                Conversation data object containing all turns.
            criteria (str):
                Description of the overall evaluation goal.
            evaluation_params (list[str]):
                List of field names to include (same semantics as turn-level).
            evaluation_steps (list[str] | None):
                Optional instructions guiding how the evaluation should proceed.
            threshold (float):
                Minimum acceptable score before the metric is considered failed.

        Returns:
            tuple[float | None, str]:
                Tuple containing (score, reason). Returns None on error.
        """
        # Convert evaluation_params to enum values if valid, otherwise use defaults
        converted_params = self._convert_evaluation_params(evaluation_params)
        if not converted_params:
            # If no valid params, use sensible defaults for conversation evaluation
            converted_params = [
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ]

        # Configure the GEval metric for conversation-level evaluation
        metric_kwargs: dict[str, Any] = {
            "name": "GEval Conversation Metric",
            "criteria": criteria,
            "evaluation_params": converted_params,
            "model": self.deepeval_llm_manager.get_llm(),
            "threshold": threshold,
            "top_logprobs": 5,  # Vertex/Gemini throws an error if over 20.
        }

        # Add evaluation steps if provided
        if evaluation_steps:
            metric_kwargs["evaluation_steps"] = evaluation_steps

        # Instantiate the GEval metric object
        metric = GEval(**metric_kwargs)

        # GEval only accepts LLMTestCase, not ConversationalTestCase
        # Aggregate conversation turns into a single test case
        conversation_input = []
        conversation_output = []

        for i, turn in enumerate(conv_data.turns, 1):
            conversation_input.append(f"Turn {i} - User: {turn.query}")
            conversation_output.append(f"Turn {i} - Assistant: {turn.response or ''}")

        # Create aggregated test case for conversation evaluation
        test_case = LLMTestCase(
            input="\n".join(conversation_input),
            actual_output="\n".join(conversation_output),
        )

        # Evaluate
        try:
            metric.measure(test_case)
            score = metric.score if metric.score is not None else 0.0
            reason = (
                str(metric.reason)
                if hasattr(metric, "reason") and metric.reason
                else "No reason provided"
            )
            return score, reason
        except Exception as e:  # pylint: disable=W0718
            logger.error(
                "GEval conversation-level evaluation failed: %s: %s",
                type(e).__name__,
                str(e),
            )
            logger.debug("Conversation turns: %d", len(conv_data.turns))
            logger.debug(
                "Test case input preview: %s...",
                test_case.input[:200] if test_case.input else "None",
            )
            return None, f"GEval evaluation error: {str(e)}"

    def _get_geval_config(
        self,
        metric_name: str,
        conv_data: Any,
        turn_data: Any | None,
        is_conversation: bool,
    ) -> dict[str, Any] | None:
        """Extract GEval configuration from metadata or registry.

         The method checks multiple sources in priority order:
            1. Turn-level metadata (runtime override)
            2. Conversation-level metadata (runtime override)
            3. Metric registry (shared, persistent YAML definitions)

         Args:
            metric_name (str):
                Name of the metric to retrieve (e.g., "completeness").
            conv_data (Any):
                The full conversation data object, which may contain
                conversation-level metadata.
            turn_data (Any | None):
                Optional turn-level data object, for per-turn metrics.
            is_conversation (bool):
                True if evaluating a conversation-level metric, False for turn-level.

        Returns:
            dict[str, Any] | None:
                The GEval configuration dictionary if found, otherwise None.
        """
        metric_key = f"geval:{metric_name}"

        # Turn level metadata override
        # Used when individual turns define custom GEval settings
        if (
            not is_conversation
            and turn_data
            and hasattr(turn_data, "turn_metrics_metadata")
            and turn_data.turn_metrics_metadata
            and metric_key in turn_data.turn_metrics_metadata
        ):
            logger.debug("Using runtime metadata for metric '%s'", metric_name)
            return turn_data.turn_metrics_metadata[metric_key]

        # Conversation-level metadata override
        # Used when the conversation defines shared GEval settings
        if (
            hasattr(conv_data, "conversation_metrics_metadata")
            and conv_data.conversation_metrics_metadata
            and metric_key in conv_data.conversation_metrics_metadata
        ):
            logger.debug("Using runtime metadata for metric '%s'", metric_name)
            return conv_data.conversation_metrics_metadata[metric_key]

        # Registry definition
        # Fallback to shared YAML registry if no runtime metadata is found
        if (
            GEvalHandler._registry
            and metric_name in GEvalHandler._registry  # pylint: disable=E1135
        ):  # pylint: disable=E1135
            logger.debug("Using registry definition for metric '%s'", metric_name)
            return GEvalHandler._registry[metric_name]  # pylint: disable=E1136

        # Config not found anywhere
        available_metrics = (
            list(GEvalHandler._registry.keys())  # pylint: disable=E1136
            if GEvalHandler._registry
            else []
        )
        logger.warning(
            "Metric '%s' not found in runtime metadata or registry. "
            "Available registry metrics: %s",
            metric_name,
            available_metrics,
        )
        return None
