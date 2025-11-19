"""Unit tests for pipeline evaluation errors module."""

from lightspeed_evaluation.core.models import EvaluationData, TurnData
from lightspeed_evaluation.pipeline.evaluation.errors import EvaluationErrorHandler


class TestEvaluationErrorHandler:
    """Unit tests for EvaluationErrorHandler."""

    def test_mark_all_metrics_as_error_with_turn_metrics(self):
        """Test marking all metrics as error with turn metrics."""
        handler = EvaluationErrorHandler()

        turn1 = TurnData(turn_id="1", query="Query 1", response="Response 1")
        turn2 = TurnData(turn_id="2", query="Query 2", response="Response 2")
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )

        resolved_turn_metrics = [
            ["ragas:faithfulness", "custom:answer_correctness"],
            ["ragas:response_relevancy"],
        ]
        resolved_conversation_metrics = []

        results = handler.mark_all_metrics_as_error(
            conv_data,
            "API Error occurred",
            resolved_turn_metrics,
            resolved_conversation_metrics,
        )

        # Should have 3 error results (2 for turn 1, 1 for turn 2)
        assert len(results) == 3

        # Check first turn metrics
        assert results[0].conversation_group_id == "test_conv"
        assert results[0].turn_id == "1"
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[0].result == "ERROR"
        assert results[0].score is None
        assert results[0].threshold is None
        assert results[0].reason == "API Error occurred"
        assert results[0].query == "Query 1"

        assert results[1].turn_id == "1"
        assert results[1].metric_identifier == "custom:answer_correctness"

        # Check second turn metrics
        assert results[2].turn_id == "2"
        assert results[2].metric_identifier == "ragas:response_relevancy"
        assert results[2].query == "Query 2"

    def test_mark_all_metrics_as_error_with_conversation_metrics(self):
        """Test marking conversation-level metrics as error."""
        handler = EvaluationErrorHandler()

        turn = TurnData(turn_id="1", query="Query", response="Response")
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        resolved_turn_metrics = [[]]
        resolved_conversation_metrics = [
            "deepeval:conversation_completeness",
            "deepeval:conversation_relevancy",
        ]

        results = handler.mark_all_metrics_as_error(
            conv_data,
            "Setup script failed",
            resolved_turn_metrics,
            resolved_conversation_metrics,
        )

        # Should have 2 conversation-level error results
        assert len(results) == 2

        # Both should be conversation-level (no turn_id)
        assert results[0].turn_id is None
        assert results[0].metric_identifier == "deepeval:conversation_completeness"
        assert results[0].result == "ERROR"
        assert results[0].reason == "Setup script failed"

        assert results[1].turn_id is None
        assert results[1].metric_identifier == "deepeval:conversation_relevancy"

    def test_mark_all_metrics_as_error_mixed(self):
        """Test marking both turn and conversation metrics as error."""
        handler = EvaluationErrorHandler()

        turn = TurnData(turn_id="1", query="Query", response="Response")
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        resolved_turn_metrics = [["ragas:faithfulness"]]
        resolved_conversation_metrics = ["deepeval:conversation_completeness"]

        results = handler.mark_all_metrics_as_error(
            conv_data,
            "Mixed error",
            resolved_turn_metrics,
            resolved_conversation_metrics,
        )

        # Should have 2 results total
        assert len(results) == 2

        # First should be turn-level
        assert results[0].turn_id == "1"
        assert results[0].metric_identifier == "ragas:faithfulness"

        # Second should be conversation-level
        assert results[1].turn_id is None
        assert results[1].metric_identifier == "deepeval:conversation_completeness"

    def test_mark_all_metrics_as_error_empty_metrics(self):
        """Test marking with no metrics to mark."""
        handler = EvaluationErrorHandler()

        turn = TurnData(turn_id="1", query="Query", response="Response")
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        resolved_turn_metrics = [[]]
        resolved_conversation_metrics = []

        results = handler.mark_all_metrics_as_error(
            conv_data, "Error", resolved_turn_metrics, resolved_conversation_metrics
        )

        # Should have no results
        assert len(results) == 0

    def test_get_error_summary_with_errors(self):
        """Test error summary after marking metrics as error."""
        handler = EvaluationErrorHandler()

        turn1 = TurnData(turn_id="1", query="Query 1", response="Response 1")
        turn2 = TurnData(turn_id="2", query="Query 2", response="Response 2")
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )

        resolved_turn_metrics = [["ragas:faithfulness"], ["custom:answer_correctness"]]
        resolved_conversation_metrics = ["deepeval:conversation_completeness"]

        handler.mark_all_metrics_as_error(
            conv_data,
            "Test error",
            resolved_turn_metrics,
            resolved_conversation_metrics,
        )

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["turn_errors"] == 2
        assert summary["conversation_errors"] == 1

    def test_multiple_error_batches(self):
        """Test accumulating errors across multiple conversations."""
        handler = EvaluationErrorHandler()

        # First conversation
        turn1 = TurnData(turn_id="1", query="Query 1", response="Response 1")
        conv1 = EvaluationData(conversation_group_id="conv1", turns=[turn1])
        handler.mark_all_metrics_as_error(
            conv1, "Error 1", [["ragas:faithfulness"]], []
        )

        # Second conversation
        turn2 = TurnData(turn_id="1", query="Query 2", response="Response 2")
        conv2 = EvaluationData(conversation_group_id="conv2", turns=[turn2])
        handler.mark_all_metrics_as_error(
            conv2, "Error 2", [["custom:answer_correctness"]], ["deepeval:completeness"]
        )

        summary = handler.get_error_summary()

        # Should accumulate errors from both conversations
        assert summary["total_errors"] == 3
        assert summary["turn_errors"] == 2
        assert summary["conversation_errors"] == 1

    def test_mark_turn_metrics_as_error(self):
        """Test marking metrics for a single turn as error."""
        handler = EvaluationErrorHandler()

        turn_data = TurnData(
            turn_id="turn1", query="Test query", response="Test response"
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn_data])

        turn_metrics = ["ragas:faithfulness", "custom:answer_correctness"]
        error_reason = "API Error: Connection timeout"

        results = handler.mark_turn_metrics_as_error(
            conv_data, 0, turn_data, turn_metrics, error_reason
        )

        # Should have 2 error results (one for each metric)
        assert len(results) == 2

        # Check first error result
        assert results[0].conversation_group_id == "test_conv"
        assert results[0].turn_id == "turn1"
        assert results[0].metric_identifier == "ragas:faithfulness"
        assert results[0].result == "ERROR"
        assert results[0].score is None
        assert results[0].threshold is None
        assert results[0].reason == error_reason
        assert results[0].query == "Test query"
        assert results[0].response == ""
        assert results[0].execution_time == 0.0

        # Check second error result
        assert results[1].conversation_group_id == "test_conv"
        assert results[1].turn_id == "turn1"
        assert results[1].metric_identifier == "custom:answer_correctness"
        assert results[1].result == "ERROR"
        assert results[1].reason == error_reason

        # Verify results are stored internally
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 2
        assert summary["turn_errors"] == 2
        assert summary["conversation_errors"] == 0

    def test_mark_cascade_failure(self):
        """Test marking remaining turns and conversation metrics as error after API failure."""
        handler = EvaluationErrorHandler()

        # Setup conversation with 3 turns
        turn1 = TurnData(turn_id="turn1", query="Query 1", response="Response 1")
        turn2 = TurnData(turn_id="turn2", query="Query 2", response="Response 2")
        turn3 = TurnData(turn_id="turn3", query="Query 3", response="Response 3")
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2, turn3]
        )

        # Resolved metrics for all turns
        resolved_turn_metrics = [
            ["ragas:faithfulness"],  # turn1
            ["custom:answer_correctness"],  # turn2
            ["ragas:response_relevancy"],  # turn3
        ]
        resolved_conversation_metrics = [
            "deepeval:conversation_completeness",
            "deepeval:conversation_relevancy",
        ]

        # API failure happens at turn 0 (first turn)
        failed_turn_idx = 0
        error_reason = "Cascade failure from turn 1 API error: Connection timeout"

        results = handler.mark_cascade_failure(
            conv_data,
            failed_turn_idx,
            resolved_turn_metrics,
            resolved_conversation_metrics,
            error_reason,
        )

        # Should have errors for:
        # - Turn 2 (1 metric) + Turn 3 (1 metric) + Conversation (2 metrics) = 4 total
        assert len(results) == 4

        # Check turn 2 error
        turn2_result = results[0]
        assert turn2_result.conversation_group_id == "test_conv"
        assert turn2_result.turn_id == "turn2"
        assert turn2_result.metric_identifier == "custom:answer_correctness"
        assert turn2_result.result == "ERROR"
        assert turn2_result.reason == error_reason

        # Check turn 3 error
        turn3_result = results[1]
        assert turn3_result.turn_id == "turn3"
        assert turn3_result.metric_identifier == "ragas:response_relevancy"
        assert turn3_result.result == "ERROR"

        # Check conversation-level errors
        conv_result1 = results[2]
        assert conv_result1.turn_id is None  # Conversation-level
        assert conv_result1.metric_identifier == "deepeval:conversation_completeness"
        assert conv_result1.result == "ERROR"

        conv_result2 = results[3]
        assert conv_result2.turn_id is None  # Conversation-level
        assert conv_result2.metric_identifier == "deepeval:conversation_relevancy"
        assert conv_result2.result == "ERROR"

        # Verify summary
        summary = handler.get_error_summary()
        assert summary["total_errors"] == 4
        assert summary["turn_errors"] == 2  # turn2 + turn3
        assert summary["conversation_errors"] == 2
