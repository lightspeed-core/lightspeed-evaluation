"""Unit tests for this process's shared run id."""

import pytest

from lightspeed_evaluation.core.output.run_id import resolve_run_id


class TestResolveRunId:
    """One run's outputs should all carry the same identifying timestamp."""

    def test_pinned_for_the_rest_of_the_process(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two calls with no env var set must agree on one id."""
        monkeypatch.delenv("LSE_RUN_ID", raising=False)

        first = resolve_run_id()
        second = resolve_run_id()

        assert first == second

    def test_explicit_env_var_is_returned_as_is(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit id is honoured rather than overwritten."""
        monkeypatch.setenv("LSE_RUN_ID", "20260101_000000")

        assert resolve_run_id() == "20260101_000000"
