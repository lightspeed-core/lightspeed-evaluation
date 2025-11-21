"""Unit tests for pipeline evaluation amender module."""

from lightspeed_evaluation.core.models import APIResponse, TurnData
from lightspeed_evaluation.core.system.exceptions import APIError
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender


class TestAPIDataAmender:
    """Unit tests for APIDataAmender."""

    def test_amend_single_turn_no_client(self):
        """Test amendment returns None when no API client is available."""
        amender = APIDataAmender(None)

        turn = TurnData(turn_id="1", query="Test query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn)

        assert error_msg is None
        assert conversation_id is None
        assert turn.response is None  # Not modified

    def test_amend_single_turn_success(self, mocker):
        """Test amending single turn data successfully."""
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

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # No error should be returned
        assert error_msg is None
        assert conversation_id == "conv_123"

        # API client should be called once
        mock_client.query.assert_called_once_with(
            query="Test query", conversation_id=None, attachments=None
        )

        # Turn data should be amended
        assert turn.response == "Generated response"
        assert turn.conversation_id == "conv_123"
        assert turn.contexts == ["Context 1", "Context 2"]

    def test_amend_single_turn_with_conversation_id(self, mocker):
        """Test amending turn with existing conversation ID."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Follow-up response",
            conversation_id="conv_123",
            contexts=["Context 3"],
            tool_calls=[],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="2", query="Follow-up query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn, "conv_123")

        # No error should be returned
        assert error_msg is None
        assert conversation_id == "conv_123"

        # API client should be called with existing conversation ID
        mock_client.query.assert_called_once_with(
            query="Follow-up query", conversation_id="conv_123", attachments=None
        )

        # Turn data should be amended
        assert turn.response == "Follow-up response"
        assert turn.conversation_id == "conv_123"
        assert turn.contexts == ["Context 3"]

    def test_amend_single_turn_with_tool_calls(self, mocker):
        """Test amending turn data with tool calls."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Tool response",
            conversation_id="conv_456",
            contexts=[],
            tool_calls=[[{"tool": "test_tool", "args": {"param": "value"}}]],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="3", query="Tool query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # No error should be returned
        assert error_msg is None
        assert conversation_id == "conv_456"

        # Turn data should be amended with tool calls
        assert turn.response == "Tool response"
        assert turn.tool_calls == [[{"tool": "test_tool", "args": {"param": "value"}}]]

    def test_amend_single_turn_with_attachments(self, mocker):
        """Test amending turn data with attachments."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Attachment response",
            conversation_id="conv_789",
            contexts=["Attachment context"],
            tool_calls=[],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(
            turn_id="4",
            query="Attachment query",
            response=None,
            attachments=["file1.txt", "file2.pdf"],
        )

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # No error should be returned
        assert error_msg is None
        assert conversation_id == "conv_789"

        # API client should be called with attachments
        mock_client.query.assert_called_once_with(
            query="Attachment query",
            conversation_id=None,
            attachments=["file1.txt", "file2.pdf"],
        )

        # Turn data should be amended
        assert turn.response == "Attachment response"
        assert turn.contexts == ["Attachment context"]

    def test_amend_single_turn_api_error(self, mocker):
        """Test handling API error during turn amendment."""
        mock_client = mocker.Mock()
        mock_client.query.side_effect = APIError("Connection failed")

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="5", query="Error query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # Error should be returned
        assert error_msg == "API Error for turn 5: Connection failed"
        assert conversation_id is None

        # Turn data should not be modified
        assert turn.response is None
        assert turn.conversation_id is None

    def test_amend_single_turn_no_contexts_in_response(self, mocker):
        """Test amending turn when API response has no contexts."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="No context response",
            conversation_id="conv_no_ctx",
            contexts=[],  # Empty contexts
            tool_calls=[],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="6", query="No context query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # No error should be returned
        assert error_msg is None
        assert conversation_id == "conv_no_ctx"

        # Turn data should be amended (contexts should remain None since API response has empty contexts)
        assert turn.response == "No context response"
        assert turn.contexts is None

    def test_amend_single_turn_no_tool_calls_in_response(self, mocker):
        """Test amending turn when API response has no tool calls."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="No tools response",
            conversation_id="conv_no_tools",
            contexts=["Context"],
            tool_calls=[],  # Empty tool calls
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="7", query="No tools query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # No error should be returned
        assert error_msg is None
        assert conversation_id == "conv_no_tools"

        # Turn data should be amended (tool_calls should remain None since API response has empty tool_calls)
        assert turn.response == "No tools response"
        assert turn.tool_calls is None
