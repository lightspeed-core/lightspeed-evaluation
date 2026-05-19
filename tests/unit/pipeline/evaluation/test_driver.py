# pylint: disable=protected-access

"""Unit tests for agent driver module."""

from typing import Any, Optional

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from lightspeed_evaluation.core.models import TurnData
from lightspeed_evaluation.core.system.exceptions import ConfigurationError
from lightspeed_evaluation.pipeline.evaluation.driver import (
    AgentDriver,
    AgentDriverRegistry,
    HttpApiDriver,
)


class TestAgentDriverRegistry:
    """Unit tests for AgentDriverRegistry."""

    def test_create_driver_success(self, mocker: MockerFixture) -> None:
        """Test creating a driver with a valid type."""
        mock_driver_cls = mocker.Mock(spec=type)
        mock_driver_instance = mocker.Mock(spec=AgentDriver)
        mock_driver_cls.return_value = mock_driver_instance

        registry = AgentDriverRegistry(drivers={"test_type": mock_driver_cls})
        driver = registry.create_driver({"type": "test_type"}, enabled=True)

        assert driver is mock_driver_instance
        mock_driver_cls.assert_called_once_with({"type": "test_type"}, enabled=True)

    def test_create_driver_missing_type(self) -> None:
        """Test creating a driver without type field raises error."""
        registry = AgentDriverRegistry()

        with pytest.raises(ConfigurationError, match="missing required 'type' field"):
            registry.create_driver({})

    def test_create_driver_unsupported_type(self) -> None:
        """Test creating a driver with unsupported type raises error."""
        registry = AgentDriverRegistry()

        with pytest.raises(ConfigurationError, match="Unsupported agent type"):
            registry.create_driver({"type": "nonexistent"})

    def test_default_registry_contains_http_api(self) -> None:
        """Test default registry includes http_api driver."""
        registry = AgentDriverRegistry()
        assert "http_api" in registry._drivers


class TestHttpApiDriver:
    """Unit tests for HttpApiDriver."""

    def test_execute_turn_delegates_to_amender(self, mocker: MockerFixture) -> None:
        """Test execute_turn delegates to internal amender."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.HttpApiAgentConfig"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIClient")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIConfig")
        mock_amender_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.APIDataAmender"
        )
        mock_amender = mock_amender_cls.return_value
        mock_amender.amend_single_turn.return_value = (None, "conv_123")

        driver = HttpApiDriver({"type": "http_api"}, enabled=True)
        turn = TurnData(turn_id="1", query="Q", response="R")
        error, conv_id = driver.execute_turn(turn, "existing_conv")

        assert error is None
        assert conv_id == "conv_123"
        mock_amender.amend_single_turn.assert_called_once_with(turn, "existing_conv")

    def test_execute_turn_returns_error(self, mocker: MockerFixture) -> None:
        """Test execute_turn propagates error from amender."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.HttpApiAgentConfig"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIClient")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIConfig")
        mock_amender_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.APIDataAmender"
        )
        mock_amender = mock_amender_cls.return_value
        mock_amender.amend_single_turn.return_value = ("API failed", None)

        driver = HttpApiDriver({"type": "http_api"}, enabled=True)
        turn = TurnData(turn_id="1", query="Q")
        error, conv_id = driver.execute_turn(turn)

        assert error == "API failed"
        assert conv_id is None

    def test_enabled_reflects_constructor_param(self, mocker: MockerFixture) -> None:
        """Test enabled property reflects constructor parameter."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.HttpApiAgentConfig"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIClient")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIConfig")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIDataAmender")

        driver = HttpApiDriver({"type": "http_api"}, enabled=True)
        assert driver.enabled is True

    def test_disabled_driver(self, mocker: MockerFixture) -> None:
        """Test disabled driver reports enabled=False and skips API client."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.HttpApiAgentConfig"
        )
        mock_api_client_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.APIClient"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIDataAmender")

        driver = HttpApiDriver({"type": "http_api"}, enabled=False)
        assert driver.enabled is False
        mock_api_client_cls.assert_not_called()

    def test_close_with_api_client(self, mocker: MockerFixture) -> None:
        """Test close cleans up API client."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.HttpApiAgentConfig"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIDataAmender")
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIConfig")

        mock_api_client_cls = mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.APIClient"
        )
        mock_api_client = mocker.Mock()
        mock_api_client_cls.return_value = mock_api_client

        driver = HttpApiDriver({"type": "http_api"}, enabled=True)
        driver.close()

        mock_api_client.close.assert_called_once()

    def test_close_without_api_client(self, mocker: MockerFixture) -> None:
        """Test close is safe when no API client exists."""
        mocker.patch(
            "lightspeed_evaluation.pipeline.evaluation.driver.HttpApiAgentConfig"
        )
        mocker.patch("lightspeed_evaluation.pipeline.evaluation.driver.APIDataAmender")

        driver = HttpApiDriver({"type": "http_api"}, enabled=False)
        driver.close()

    def test_validate_config_invalid(self) -> None:
        """Test validate_config rejects invalid configuration."""
        with pytest.raises(ValidationError):
            HttpApiDriver(
                {"type": "http_api", "invalid_field_only": True}, enabled=True
            )


class TestAgentDriverBase:
    """Unit tests for AgentDriver base class defaults."""

    def test_enabled_default_is_true(self) -> None:
        """Test base class enabled property defaults to True."""

        class StubDriver(AgentDriver):
            """Minimal driver for testing."""

            def execute_turn(
                self, turn_data: TurnData, conversation_id: Optional[str] = None
            ) -> tuple[Optional[str], Optional[str]]:
                return (None, None)

            def validate_config(self, config: dict[str, Any]) -> Any:
                return config

        driver = StubDriver({})
        assert driver.enabled is True

    def test_enabled_false_at_construction(self) -> None:
        """Test base class enabled=False is propagated."""

        class StubDriver(AgentDriver):
            """Minimal driver for testing."""

            def execute_turn(
                self, turn_data: TurnData, conversation_id: Optional[str] = None
            ) -> tuple[Optional[str], Optional[str]]:
                return (None, None)

            def validate_config(self, config: dict[str, Any]) -> Any:
                return config

        driver = StubDriver({}, enabled=False)
        assert driver.enabled is False

    def test_close_is_noop(self) -> None:
        """Test base class close is a no-op."""

        class StubDriver(AgentDriver):
            """Minimal driver for testing."""

            def execute_turn(
                self, turn_data: TurnData, conversation_id: Optional[str] = None
            ) -> tuple[Optional[str], Optional[str]]:
                return (None, None)

            def validate_config(self, config: dict[str, Any]) -> Any:
                return config

        driver = StubDriver({})
        driver.close()
