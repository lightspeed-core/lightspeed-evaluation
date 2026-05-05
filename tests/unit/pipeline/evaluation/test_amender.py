"""Unit tests for pipeline evaluation amender module."""

from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import APIResponse, TurnData
from lightspeed_evaluation.core.system.exceptions import APIError
from lightspeed_evaluation.pipeline.evaluation.amender import APIDataAmender


class TestAPIDataAmender:
    """Unit tests for APIDataAmender."""

    def test_amend_single_turn_no_client(self) -> None:
        """Test amendment returns None when no API client is available."""
        amender = APIDataAmender(None)

        turn = TurnData(turn_id="1", query="Test query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn)

        assert error_msg is None
        assert conversation_id is None
        assert turn.response is None  # Not modified

    def test_amend_single_turn_success(self, mocker: MockerFixture) -> None:
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
            query="Test query",
            conversation_id=None,
            attachments=None,
            extra_request_params=None,
        )

        # Turn data should be amended
        assert turn.response == "Generated response"
        assert turn.conversation_id == "conv_123"
        assert turn.contexts == ["Context 1", "Context 2"]

    def test_amend_single_turn_with_conversation_id(
        self, mocker: MockerFixture
    ) -> None:
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
            query="Follow-up query",
            conversation_id="conv_123",
            attachments=None,
            extra_request_params=None,
        )

        # Turn data should be amended
        assert turn.response == "Follow-up response"
        assert turn.conversation_id == "conv_123"
        assert turn.contexts == ["Context 3"]

    def test_amend_single_turn_with_tool_calls(self, mocker: MockerFixture) -> None:
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

    def test_amend_single_turn_with_attachments(self, mocker: MockerFixture) -> None:
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
            extra_request_params=None,
        )

        # Turn data should be amended
        assert turn.response == "Attachment response"
        assert turn.contexts == ["Attachment context"]

    def test_amend_single_turn_api_error(self, mocker: MockerFixture) -> None:
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

    def test_amend_single_turn_no_contexts_in_response(
        self, mocker: MockerFixture
    ) -> None:
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

        # Turn data should be amended (contexts should remain None since API response
        # has empty contexts)
        assert turn.response == "No context response"
        assert turn.contexts is None

    def test_amend_single_turn_no_tool_calls_in_response(
        self, mocker: MockerFixture
    ) -> None:
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

        # Turn data should be amended (tool_calls should remain None since API response
        # has empty tool_calls)
        assert turn.response == "No tools response"
        assert turn.tool_calls is None

    def test_amend_single_turn_with_extra_request_params(
        self, mocker: MockerFixture
    ) -> None:
        """Test amending turn data passes extra_request_params to API client."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Troubleshoot response",
            conversation_id="conv_extra",
            contexts=[],
            tool_calls=[],
        )
        mock_client.query.return_value = api_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(
            turn_id="8",
            query="Troubleshoot query",
            response=None,
            extra_request_params={"mode": "troubleshooting"},
        )

        error_msg, conversation_id = amender.amend_single_turn(turn)

        assert error_msg is None
        assert conversation_id == "conv_extra"

        mock_client.query.assert_called_once_with(
            query="Troubleshoot query",
            conversation_id=None,
            attachments=None,
            extra_request_params={"mode": "troubleshooting"},
        )

    def test_amend_single_turn_measures_agent_latency(
        self, mocker: MockerFixture
    ) -> None:
        """Test that agent_latency is measured for actual API calls (with tokens)."""
        mock_client = mocker.Mock()
        api_response = APIResponse(
            response="Test response",
            conversation_id="conv_latency",
            contexts=[],
            tool_calls=[],
            input_tokens=100,
            output_tokens=50,
        )
        mock_client.query.return_value = api_response

        # Mock time.perf_counter to return deterministic timing values
        mocker.patch(
            "time.perf_counter",
            side_effect=[1.0, 1.5],  # Start: 1.0, End: 1.5 → latency = 0.5
        )

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="9", query="Latency test query", response=None)

        # Initial agent_latency should be 0 (default)
        assert turn.agent_latency == 0

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # No error should be returned
        assert error_msg is None
        assert conversation_id == "conv_latency"

        # agent_latency should be measured (exactly 0.5s) for actual API call
        assert turn.agent_latency == 0.5
        assert turn.api_input_tokens == 100
        assert turn.api_output_tokens == 50

    def test_amend_single_turn_no_agent_latency_when_no_client(self) -> None:
        """Test that agent_latency is NOT measured when API client is None (api_enabled=False)."""
        amender = APIDataAmender(None)

        turn = TurnData(turn_id="10", query="No API query", response=None)

        # Initial agent_latency should be 0 (default)
        assert turn.agent_latency == 0

        error_msg, conversation_id = amender.amend_single_turn(turn)

        # No error should be returned
        assert error_msg is None
        assert conversation_id is None

        # agent_latency should remain 0 since no API call was made
        assert turn.agent_latency == 0

    def test_amend_single_turn_no_latency_for_cached_responses(
        self, mocker: MockerFixture
    ) -> None:
        """Test that agent_latency is 0 for cached responses (zero tokens)."""
        mock_client = mocker.Mock()
        # Cached responses have zero tokens (set by cache retrieval logic)
        cached_response = APIResponse(
            response="Cached response",
            conversation_id="conv_cached",
            contexts=[],
            tool_calls=[],
            input_tokens=0,
            output_tokens=0,
        )
        mock_client.query.return_value = cached_response

        amender = APIDataAmender(mock_client)

        turn = TurnData(turn_id="11", query="Cached query", response=None)

        error_msg, conversation_id = amender.amend_single_turn(turn)

        assert error_msg is None
        assert conversation_id == "conv_cached"

        # Cached response should have zero latency (no actual API call)
        assert turn.agent_latency == 0
        assert turn.api_input_tokens == 0
        assert turn.api_output_tokens == 0
