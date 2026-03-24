"""Unit tests for storage exceptions."""

import pytest

from lightspeed_evaluation.core.storage import StorageError
from lightspeed_evaluation.core.system.exceptions import EvaluationError


class TestStorageError:
    """Tests for StorageError exception."""

    def test_inherits_from_evaluation_error(self) -> None:
        """Test that StorageError inherits from EvaluationError."""
        error = StorageError("test error")
        assert isinstance(error, EvaluationError)
        assert isinstance(error, Exception)

    def test_message_only(self) -> None:
        """Test creating error with message only."""
        error = StorageError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.backend_name is None

    def test_with_backend_name(self) -> None:
        """Test creating error with backend name."""
        error = StorageError("Connection failed", backend_name="sqlite")
        assert str(error) == "Connection failed"
        assert error.backend_name == "sqlite"

    def test_catchable_by_evaluation_error(self) -> None:
        """Test that StorageError can be caught by EvaluationError."""
        with pytest.raises(EvaluationError):
            raise StorageError("test")
