"""Tests for trace value coercion helpers."""

from datetime import UTC, datetime
from typing import Any

import pytest

from lightspeed_evaluation.core.tracing._helpers import _as_datetime


class TestAsDatetime:
    """UTC normalization for all supported time representations."""

    def test_naive_iso_assumes_utc(self) -> None:
        """ISO strings without timezone are treated as UTC."""
        result = _as_datetime("2024-01-01T12:00:00")
        assert result == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)

    def test_zulu_and_offset_iso_normalize_to_utc(self) -> None:
        """Z and explicit offsets both land on UTC-aware datetimes."""
        assert _as_datetime("2024-01-01T12:00:00Z") == datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=UTC
        )
        assert _as_datetime("2024-01-01T12:00:00+05:30") == datetime(
            2024, 1, 1, 6, 30, 0, tzinfo=UTC
        )

    def test_naive_datetime_and_epoch_are_utc(self) -> None:
        """Naive datetime objects and epoch values are UTC-aware."""
        naive = datetime(2024, 1, 1, 12, 0, 0)
        assert _as_datetime(naive) == datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        epoch = _as_datetime(1_700_000_000)
        assert epoch is not None
        assert epoch.tzinfo == UTC


@pytest.mark.parametrize("value", [float("inf"), float("nan"), 10**100, "invalid"])
def test_invalid_timestamps_do_not_crash(value: Any) -> None:
    """Unrepresentable optional timestamps normalize to missing."""
    assert _as_datetime(value) is None
