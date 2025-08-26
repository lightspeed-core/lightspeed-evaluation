"""
Evaluation Engine - Main evaluation controller.

Controls the evaluation flow through conversations & turns
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

from .core import (
    ConfigLoader,
    DataValidator,
    EvaluationData,
    EvaluationResult,
    TurnData,
)
from .llm_managers.llm_manager import LLMManager
from .metrics.custom_metrics import CustomMetrics
from .metrics.deepeval_metrics import DeepEvalMetrics
from .metrics.ragas_metrics import RagasMetrics
from .output.utils import EvaluationScope


class EvaluationRequest:
    """Evaluation request containing conversation data and metric information."""

    def __init__(self, conv_data: EvaluationData, metric_identifier: str):
        """Initialize evaluation request with conversation data and metric identifier."""
        self.conv_data = conv_data
        self.metric_identifier = metric_identifier
        self.is_conversation = False
        self.turn_idx: Optional[int] = None
        self.turn_data: Optional[TurnData] = None
        self.turn_id: Optional[int] = None

    @classmethod
    def for_turn(
        cls,
        conv_data: EvaluationData,
        metric_identifier: str,
        turn_idx: int,
        turn_data: TurnData,
    ) -> "EvaluationRequest":
        """Create request for turn-level evaluation."""
        request = cls(conv_data, metric_identifier)
        request.is_conversation = False
        request.turn_idx = turn_idx
        request.turn_data = turn_data
        request.turn_id = turn_data.turn_id if turn_data.turn_id else turn_idx + 1
        return request

    @classmethod
    def for_conversation(
        cls, conv_data: EvaluationData, metric_identifier: str
    ) -> "EvaluationRequest":
        """Create request for conversation-level evaluation."""
        request = cls(conv_data, metric_identifier)
        request.is_conversation = True
        return request

    def get_turn_display_info(self) -> str:
        """Get displayable information about the turn request."""
        if self.is_conversation:
            return f"Conversation: {self.conv_data.conversation_group_id}"
        return f"TurnData {self.turn_id}: {self.conv_data.conversation_group_id}"

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation request."""
        return {
            "metric": self.metric_identifier,
            "conversation_id": self.conv_data.conversation_group_id,
            "is_conversation_level": self.is_conversation,
            "turn_id": self.turn_id if not self.is_conversation else None,
        }


class MetricsManager:
    """Manages all metric evaluation handlers."""

    def __init__(self, llm_manager: LLMManager):
        """Initialize with LLM manager."""
        self.ragas_metrics = RagasMetrics(llm_manager)
        self.deepeval_metrics = DeepEvalMetrics(llm_manager)
        self.custom_metrics = CustomMetrics(llm_manager)

        # Metric routing map
        self.handlers: Dict[
            str, Union[RagasMetrics, DeepEvalMetrics, CustomMetrics]
        ] = {
            "ragas": self.ragas_metrics,
            "deepeval": self.deepeval_metrics,
            "custom": self.custom_metrics,
        }

    def evaluate_metric(
        self, framework: str, metric_name: str, request: EvaluationRequest
    ) -> Tuple[Optional[float], str]:
        """Route evaluation to appropriate handler."""
        if framework in self.handlers:
            # Create shared EvaluationScope
            evaluation_scope = EvaluationScope(
                turn_idx=request.turn_idx,
                turn_data=request.turn_data,
                is_conversation=request.is_conversation,
            )
            return self.handlers[framework].evaluate(
                metric_name,
                request.conv_data,
                evaluation_scope,
            )
        return None, f"Unsupported framework: {framework}"

    def get_supported_frameworks(self) -> List[str]:
        """Get list of supported evaluation frameworks."""
        return list(self.handlers.keys())


class EvaluationEngine:
    """
    Main evaluation engine - orchestrates the evaluation process.

    Responsibilities:
    - Data validation
    - Metric routing and evaluation
    - Result collection
    - Status determination (PASS/FAIL/ERROR)
    """

    def __init__(self, config_loader: ConfigLoader):
        """Initialize evaluation engine with config."""
        self.config_loader = config_loader
        self.data_validator = DataValidator()
        self.results: List[EvaluationResult] = []

        # Initialize LLM Manager and metrics
        system_config_dict = config_loader.get_llm_config_dict()
        llm_manager = LLMManager.from_system_config(system_config_dict)
        self.metrics_manager = MetricsManager(llm_manager)

        print("✅ Evaluation Engine initialized")

    def validate_data(self, evaluation_data: List[EvaluationData]) -> bool:
        """Validate evaluation data using data validator."""
        return self.data_validator.validate_evaluation_data(evaluation_data)

    def run_evaluation(
        self, evaluation_data: List[EvaluationData]
    ) -> List[EvaluationResult]:
        """
        Run complete evaluation pipeline.

        Args:
            evaluation_data: List of conversation data to evaluate

        Returns:
            List of evaluation results
        """
        print("🚀 Starting evaluation...")
        self.results = []

        # Step 1: Validate data
        print("\n1️⃣ Validating data...")
        if not self.data_validator.validate_evaluation_data(evaluation_data):
            raise ValueError("Data validation failed. Cannot proceed with evaluation.")

        # Step 2: Process each conversation
        print("\n2️⃣ Processing conversations...")
        for conv_data in evaluation_data:
            self._process_conversation(conv_data)

        print(f"\n✅ Evaluation complete: {len(self.results)} results generated")
        return self.results

    def _process_conversation(self, conv_data: EvaluationData) -> None:
        """Process single conversation - handle turn and conversation level metrics."""
        print(f"\n📋 Evaluating: {conv_data.conversation_group_id}")

        # Skip if no metrics specified at any level
        if not conv_data.turn_metrics and not conv_data.conversation_metrics:
            print("⏭️ No metrics specified, skipping")
            return

        # Process turn-level metrics
        if conv_data.turn_metrics:
            print(f"🔄 Turn-level metrics: {conv_data.turn_metrics}")
            for turn_idx, turn_data in enumerate(conv_data.turns):
                self._evaluate_turn(conv_data, turn_idx, turn_data)

        # Process conversation-level metrics
        if conv_data.conversation_metrics:
            print(f"🗣️ Conversation-level metrics: {conv_data.conversation_metrics}")
            self._evaluate_conversation(conv_data)

    def _evaluate_turn(
        self, conv_data: EvaluationData, turn_idx: int, turn_data: TurnData
    ) -> None:
        """Evaluate single turn with specified turn metrics."""
        for metric_identifier in conv_data.turn_metrics:
            request = EvaluationRequest.for_turn(
                conv_data, metric_identifier, turn_idx, turn_data
            )
            result = self._evaluate_metric(request)
            if result:
                self.results.append(result)

    def _evaluate_conversation(self, conv_data: EvaluationData) -> None:
        """Evaluate conversation-level metrics."""
        for metric_identifier in conv_data.conversation_metrics:
            request = EvaluationRequest.for_conversation(conv_data, metric_identifier)
            result = self._evaluate_metric(request)
            if result:
                self.results.append(result)

    def _evaluate_metric(
        self, request: EvaluationRequest
    ) -> Optional[EvaluationResult]:
        """
        Evaluate single metric using context.

        Returns:
            EvaluationResult or None if evaluation fails
        """
        start_time = time.time()

        try:
            framework, metric_name = request.metric_identifier.split(":", 1)

            # Get threshold
            threshold = self._get_effective_threshold(
                request.conv_data, request.metric_identifier, request.is_conversation
            )

            print(f"    {request.metric_identifier} (threshold: {threshold})")

            # Route to metrics manager
            score, reason = self.metrics_manager.evaluate_metric(
                framework, metric_name, request
            )

            # Determine result status
            if score is None:
                result_status = "ERROR"
                print(f"      ❌ ERROR: {reason}")
            else:
                result_status = self._determine_status(score, threshold)
                status_emoji = (
                    "✅"
                    if result_status == "PASS"
                    else "❌" if result_status == "FAIL" else "⚠️"
                )
                print(f"      {status_emoji} {result_status}: {score:.3f}")

            # Create result
            execution_time = time.time() - start_time

            return EvaluationResult(
                conversation_group_id=request.conv_data.conversation_group_id,
                turn_id=request.turn_id,
                metric_identifier=request.metric_identifier,
                result=result_status,
                score=score,
                threshold=threshold,
                reason=reason,
                query=request.turn_data.query if request.turn_data else "",
                response=request.turn_data.response if request.turn_data else "",
                execution_time=execution_time,
            )

        except (ValueError, KeyError, AttributeError) as e:
            execution_time = time.time() - start_time
            error_reason = f"Evaluation failed: {str(e)}"
            print(f"      ❌ ERROR: {error_reason}")

            return EvaluationResult(
                conversation_group_id=request.conv_data.conversation_group_id,
                turn_id=request.turn_id,
                metric_identifier=request.metric_identifier,
                result="ERROR",
                score=None,
                threshold=None,
                reason=error_reason,
                query=request.turn_data.query if request.turn_data else "",
                response=request.turn_data.response if request.turn_data else "",
                execution_time=execution_time,
            )

    def _get_effective_threshold(
        self, conv_data: EvaluationData, metric_identifier: str, is_conversation: bool
    ) -> Optional[float]:
        """Get effective threshold for metric (conversation-specific or system default)."""
        # Check conversation-specific metadata first
        if is_conversation:
            metadata = conv_data.conversation_metrics_metadata.get(
                metric_identifier, {}
            )
        else:
            metadata = conv_data.turn_metrics_metadata.get(metric_identifier, {})

        if "threshold" in metadata:
            return metadata["threshold"]

        # Fall back to system defaults
        system_config = self.config_loader.system_config
        if not system_config:
            return None

        if is_conversation:
            default_metadata = (
                system_config.default_conversation_metrics_metadata or {}
            ).get(metric_identifier, {})
        else:
            default_metadata = (system_config.default_turn_metrics_metadata or {}).get(
                metric_identifier, {}
            )

        return default_metadata.get("threshold")

    def _determine_status(self, score: float, threshold: Optional[float]) -> str:
        """Determine evaluation status based on score and threshold."""
        if threshold is None:
            return "PASS"  # No threshold means always pass
        return "PASS" if score >= threshold else "FAIL"
