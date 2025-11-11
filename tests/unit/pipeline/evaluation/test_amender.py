"""Unit tests for pipeline evaluation amender module."""

from lightspeed_evaluation.core.models import APIResponse, EvaluationData, TurnData
from lightspeed_evaluation.core.system.exceptions import APIError
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender


class TestAPIDataAmender:
    """Unit tests for APIDataAmender."""

    def test_amend_conversation_data_no_client(self):
        """Test amendment returns None when no API client is available."""
        amender = APIDataAmender(None)

        turn = TurnData(turn_id="1", query="Test query", response=None)
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = amender.amend_conversation_data(conv_data)

        assert result is None
        assert turn.response is None  # Not modified

    def test_amend_conversation_data_single_turn(self, mocker):
        """Test amending conversation data with single turn."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Generated response",
            conversation_id="conv_123",
            contexts=["Context 1", "Context 2"],
            tool_calls=[],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="1", query="Test query", response=None)
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = amender.amend_conversation_data(conv_data)

        # No error should be returned
        assert result is None

        # API client should be called once
        mock_client.query.assert_called_once_with(
            query="Test query", conversation_id=None, attachments=None
        )

        # Turn data should be amended
        assert turn.response == "Generated response"
        assert turn.conversation_id == "conv_123"
        assert turn.contexts == ["Context 1", "Context 2"]

    def test_amend_conversation_data_multiple_turns(self, mocker):
        """Test amending conversation with multiple turns maintains conversation_id."""
        mock_client = mocker.Mock()

        # First turn response
        response1 = APIResponse(
            response="Response 1",
            conversation_id="conv_123",
            contexts=["Context 1"],
            tool_calls=[],
        )

        # Second turn response (same conversation)
        response2 = APIResponse(
            response="Response 2",
            conversation_id="conv_123",
            contexts=["Context 2"],
            tool_calls=[],
        )

        mock_client.query.side_effect = [response1, response2]

        amender = APIDataAmender(mock_client)

        turn1 = TurnData(turn_id="1", query="Query 1", response=None)
        turn2 = TurnData(turn_id="2", query="Query 2", response=None)
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )

        result = amender.amend_conversation_data(conv_data)

        assert result is None

        # Should be called twice
        assert mock_client.query.call_count == 2

        # First call without conversation_id
        mock_client.query.assert_any_call(
            query="Query 1", conversation_id=None, attachments=None
        )

        # Second call with conversation_id from first response
        mock_client.query.assert_any_call(
            query="Query 2", conversation_id="conv_123", attachments=None
        )

        # Both turns should be amended
        assert turn1.response == "Response 1"
        assert turn1.conversation_id == "conv_123"
        assert turn2.response == "Response 2"
        assert turn2.conversation_id == "conv_123"

    def test_amend_conversation_data_with_tool_calls(self, mocker):
        """Test amending turn data with tool calls."""
        mock_client = mocker.Mock()

        tool_calls = [
            [{"tool_name": "search", "arguments": {"query": "test"}}],
            [{"tool_name": "calculator", "arguments": {"expr": "2+2"}}],
        ]

        api_response = APIResponse(
            response="Used tools",
            conversation_id="conv_123",
            contexts=["Context"],
            tool_calls=tool_calls,
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="1", query="Query with tools", response=None)
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = amender.amend_conversation_data(conv_data)

        assert result is None
        assert turn.tool_calls == tool_calls

    def test_amend_conversation_data_with_attachments(self, mocker):
        """Test amending turn with attachments."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Response with attachments",
            conversation_id="conv_123",
            contexts=["Context"],
            tool_calls=[],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(
            turn_id="1",
            query="Query",
            response=None,
            attachments=["file1.txt", "file2.pdf"],
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = amender.amend_conversation_data(conv_data)

        assert result is None

        # Should pass attachments to API
        mock_client.query.assert_called_once_with(
            query="Query", conversation_id=None, attachments=["file1.txt", "file2.pdf"]
        )

    def test_amend_conversation_data_api_error_first_turn(self, mocker):
        """Test API error on first turn returns error message."""
        mock_client = mocker.Mock()
        mock_client.query.side_effect = APIError("Connection failed")

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="1", query="Query", response=None)
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = amender.amend_conversation_data(conv_data)

        # Should return error message
        assert result is not None
        assert "API Error for turn 1" in result
        assert "Connection failed" in result

    def test_amend_conversation_data_api_error_second_turn(self, mocker):
        """Test API error on second turn after first succeeds."""
        mock_client = mocker.Mock()

        # First turn succeeds
        response1 = APIResponse(
            response="Response 1",
            conversation_id="conv_123",
            contexts=["Context"],
            tool_calls=[],
        )

        # Second turn fails
        mock_client.query.side_effect = [response1, APIError("Rate limit exceeded")]

        amender = APIDataAmender(mock_client)

        turn1 = TurnData(turn_id="1", query="Query 1", response=None)
        turn2 = TurnData(turn_id="2", query="Query 2", response=None)
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )

        result = amender.amend_conversation_data(conv_data)

        # Should return error message for turn 2
        assert result is not None
        assert "API Error for turn 2" in result
        assert "Rate limit exceeded" in result

        # First turn should still be amended
        assert turn1.response == "Response 1"
        # Second turn should not be amended
        assert turn2.response is None

    def test_amend_conversation_data_no_contexts_in_response(self, mocker):
        """Test amending when API response has no contexts."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Response without contexts",
            conversation_id="conv_123",
            contexts=[],
            tool_calls=[],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(
            turn_id="1", query="Query", response=None, contexts=["Original context"]
        )
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        result = amender.amend_conversation_data(conv_data)

        assert result is None
        assert turn.response == "Response without contexts"
        # Contexts should remain unchanged when API returns empty list
        assert turn.contexts == ["Original context"]

    def test_get_amendment_summary_with_client(self, mocker):
        """Test getting amendment summary with API client."""
        mock_client = mocker.Mock()
        amender = APIDataAmender(mock_client)

        turn1 = TurnData(turn_id="1", query="Q1", response="R1")
        turn2 = TurnData(turn_id="2", query="Q2", response=None)
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )

        summary = amender.get_amendment_summary(conv_data)

        assert summary["conversation_group_id"] == "test_conv"
        assert summary["total_turns"] == 2
        assert summary["api_enabled"] is True
        assert summary["turns_with_existing_data"] == 1

    def test_get_amendment_summary_without_client(self):
        """Test getting amendment summary without API client."""
        amender = APIDataAmender(None)

        turn = TurnData(turn_id="1", query="Query", response=None)
        conv_data = EvaluationData(conversation_group_id="test_conv", turns=[turn])

        summary = amender.get_amendment_summary(conv_data)

        assert summary["conversation_group_id"] == "test_conv"
        assert summary["total_turns"] == 1
        assert summary["api_enabled"] is False
        assert summary["turns_with_existing_data"] == 0

    def test_get_amendment_summary_with_tool_calls(self, mocker):
        """Test summary counts turns with tool calls as having existing data."""
        mock_client = mocker.Mock()
        amender = APIDataAmender(mock_client)

        turn1 = TurnData(
            turn_id="1",
            query="Q1",
            response=None,
            tool_calls=[[{"tool_name": "search", "arguments": {}}]],
        )
        turn2 = TurnData(turn_id="2", query="Q2", response=None)
        conv_data = EvaluationData(
            conversation_group_id="test_conv", turns=[turn1, turn2]
        )

        summary = amender.get_amendment_summary(conv_data)

        assert summary["turns_with_existing_data"] == 1  # turn1 has tool_calls
