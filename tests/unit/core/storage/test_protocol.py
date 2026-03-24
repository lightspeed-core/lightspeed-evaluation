"""Unit tests for storage protocol and RunInfo."""

from datetime import datetime, timezone

from lightspeed_evaluation.core.storage import RunInfo


class TestRunInfo:
    """Tests for RunInfo dataclass."""

    def test_generates_unique_run_id(self) -> None:
        """Test that each RunInfo gets a unique run_id."""
        run1 = RunInfo()
        run2 = RunInfo()
        assert run1.run_id != run2.run_id

    def test_with_name(self) -> None:
        """Test creating RunInfo with a name."""
        run_info = RunInfo(name="test_evaluation")
        assert run_info.name == "test_evaluation"
        assert run_info.run_id  # Should have a generated ID

    def test_sets_timestamp(self) -> None:
        """Test that RunInfo sets a timestamp."""
        before = datetime.now(timezone.utc)
        run_info = RunInfo()
        after = datetime.now(timezone.utc)

        assert before <= run_info.started_at <= after

    def test_default_values(self) -> None:
        """Test default values."""
        run_info = RunInfo()
        assert run_info.run_id  # Auto-generated
        assert run_info.name == ""
        assert run_info.started_at is not None

    def test_run_id_is_uuid_format(self) -> None:
        """Test that run_id is in UUID format (8-4-4-4-12)."""
        run_info = RunInfo()
        parts = run_info.run_id.split("-")
        assert len(parts) == 5
        assert [len(p) for p in parts] == [8, 4, 4, 4, 12]
