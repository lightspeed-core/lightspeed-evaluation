"""Unit tests for litellm_patch vertex override support."""

import asyncio
import threading
from typing import Any, Callable

import pytest
from pytest_mock import MockerFixture
import litellm

from lightspeed_evaluation.core.llm import litellm_patch
from lightspeed_evaluation.core.llm.litellm_patch import (
    _vertex_override,
    _vertex_override_async,
)


class TestVertexOverrideContextManager:
    """Tests for the _vertex_override context manager."""

    def test_no_vertex_params_is_noop(self) -> None:
        """Test that _vertex_override is a no-op when no vertex params present."""
        kwargs: dict[str, Any] = {"model": "gpt-4", "temperature": 0.5}
        original_kwargs = dict(kwargs)

        with _vertex_override(kwargs):
            pass

        assert kwargs == original_kwargs

    def test_vertex_location_set_and_restored(self) -> None:
        """Test that vertex_location is set on litellm module and restored after."""
        old_value = getattr(litellm, "vertex_location", None)
        kwargs: dict[str, Any] = {"vertex_location": "us-central1"}

        with _vertex_override(kwargs):
            assert litellm.vertex_location == "us-central1"
            assert "vertex_location" not in kwargs

        assert getattr(litellm, "vertex_location", None) == old_value

    def test_vertex_project_set_and_restored(self) -> None:
        """Test that vertex_project is set on litellm module and restored after."""
        old_value = getattr(litellm, "vertex_project", None)
        kwargs: dict[str, Any] = {"vertex_project": "my-project"}

        with _vertex_override(kwargs):
            assert litellm.vertex_project == "my-project"
            assert "vertex_project" not in kwargs

        assert getattr(litellm, "vertex_project", None) == old_value

    def test_both_params_set_and_restored(self) -> None:
        """Test that both vertex params are set and restored."""
        old_location = getattr(litellm, "vertex_location", None)
        old_project = getattr(litellm, "vertex_project", None)
        kwargs: dict[str, Any] = {
            "vertex_location": "europe-west1",
            "vertex_project": "my-project",
            "temperature": 0.5,
        }

        with _vertex_override(kwargs):
            assert litellm.vertex_location == "europe-west1"
            assert litellm.vertex_project == "my-project"
            assert "vertex_location" not in kwargs
            assert "vertex_project" not in kwargs
            assert kwargs == {"temperature": 0.5}

        assert getattr(litellm, "vertex_location", None) == old_location
        assert getattr(litellm, "vertex_project", None) == old_project

    def test_params_restored_on_exception(self) -> None:
        """Test that vertex params are restored even when an exception occurs."""
        old_location = getattr(litellm, "vertex_location", None)
        kwargs: dict[str, Any] = {"vertex_location": "us-east1"}

        with pytest.raises(ValueError, match="test error"):
            with _vertex_override(kwargs):
                assert litellm.vertex_location == "us-east1"
                raise ValueError("test error")

        assert getattr(litellm, "vertex_location", None) == old_location

    def test_lock_acquired_without_vertex_params(self, mocker: MockerFixture) -> None:
        """Test that the lock is acquired even when no vertex params are present."""
        mock_lock = mocker.patch.object(litellm_patch, "litellm_state_lock")
        kwargs: dict[str, Any] = {"temperature": 0.5}

        with _vertex_override(kwargs):
            pass

        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()

    def test_lock_acquired_with_vertex_params(self, mocker: MockerFixture) -> None:
        """Test that the lock is acquired when vertex params are present."""
        mock_lock = mocker.MagicMock()
        mocker.patch.object(litellm_patch, "litellm_state_lock", mock_lock)
        kwargs: dict[str, Any] = {"vertex_location": "us-central1"}

        with _vertex_override(kwargs):
            pass

        mock_lock.__enter__.assert_called_once()
        mock_lock.__exit__.assert_called_once()


class TestVertexOverrideAsyncContextManager:
    """Tests for the _vertex_override_async async context manager."""

    @pytest.mark.asyncio
    async def test_no_vertex_params_is_noop(self) -> None:
        """Test that _vertex_override_async is a no-op when no vertex params present."""
        kwargs: dict[str, Any] = {"model": "gpt-4", "temperature": 0.5}
        original_kwargs = dict(kwargs)

        async with _vertex_override_async(kwargs):
            pass

        assert kwargs == original_kwargs

    @pytest.mark.asyncio
    async def test_vertex_location_set_and_restored(self) -> None:
        """Test that vertex_location is set on litellm module and restored after."""
        old_value = getattr(litellm, "vertex_location", None)
        kwargs: dict[str, Any] = {"vertex_location": "us-central1"}

        async with _vertex_override_async(kwargs):
            assert litellm.vertex_location == "us-central1"
            assert "vertex_location" not in kwargs

        assert getattr(litellm, "vertex_location", None) == old_value

    @pytest.mark.asyncio
    async def test_both_params_set_and_restored(self) -> None:
        """Test that both vertex params are set and restored."""
        old_location = getattr(litellm, "vertex_location", None)
        old_project = getattr(litellm, "vertex_project", None)
        kwargs: dict[str, Any] = {
            "vertex_location": "europe-west1",
            "vertex_project": "my-project",
            "temperature": 0.5,
        }

        async with _vertex_override_async(kwargs):
            assert litellm.vertex_location == "europe-west1"
            assert litellm.vertex_project == "my-project"
            assert "vertex_location" not in kwargs
            assert "vertex_project" not in kwargs
            assert kwargs == {"temperature": 0.5}

        assert getattr(litellm, "vertex_location", None) == old_location
        assert getattr(litellm, "vertex_project", None) == old_project

    @pytest.mark.asyncio
    async def test_params_restored_on_exception(self) -> None:
        """Test that vertex params are restored even when an exception occurs."""
        old_location = getattr(litellm, "vertex_location", None)
        kwargs: dict[str, Any] = {"vertex_location": "us-east1"}

        with pytest.raises(ValueError, match="test error"):
            async with _vertex_override_async(kwargs):
                assert litellm.vertex_location == "us-east1"
                raise ValueError("test error")

        assert getattr(litellm, "vertex_location", None) == old_location

    @pytest.mark.asyncio
    async def test_threading_lock_used_with_vertex_params(
        self, mocker: MockerFixture
    ) -> None:
        """Test that litellm_state_lock is acquired and held across yield."""
        mock_lock = mocker.MagicMock()
        mocker.patch.object(litellm_patch, "litellm_state_lock", mock_lock)
        kwargs: dict[str, Any] = {"vertex_location": "us-central1"}

        async with _vertex_override_async(kwargs):
            mock_lock.acquire.assert_called_once()
            mock_lock.release.assert_not_called()

        mock_lock.release.assert_called_once()

    @pytest.mark.asyncio
    async def test_threading_lock_not_used_without_vertex_params(
        self, mocker: MockerFixture
    ) -> None:
        """Test that litellm_state_lock is not acquired when no vertex params."""
        mock_lock = mocker.MagicMock()
        mocker.patch.object(litellm_patch, "litellm_state_lock", mock_lock)
        kwargs: dict[str, Any] = {"temperature": 0.5}

        async with _vertex_override_async(kwargs):
            pass

        mock_lock.acquire.assert_not_called()


class TestCompletionWithVertexOverride:
    """Test litellm.completion integration with vertex override."""

    def test_completion_with_vertex_location(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that vertex_location is handled during completion calls."""
        mock_completion = mocker.patch(f"{litellm_patch.__name__}._original_completion")
        mock_completion.return_value = mock_judge_llm_response(
            prompt_tokens=10, completion_tokens=5, cache_hit=False, content="ok"
        )

        old_location = getattr(litellm, "vertex_location", None)

        litellm.completion(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            vertex_location="us-central1",
        )

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert "vertex_location" not in call_kwargs
        assert getattr(litellm, "vertex_location", None) == old_location

    def test_completion_without_vertex_params_unchanged(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that completion works normally without vertex params."""
        mock_completion = mocker.patch(f"{litellm_patch.__name__}._original_completion")
        mock_completion.return_value = mock_judge_llm_response(
            prompt_tokens=10, completion_tokens=5, cache_hit=False, content="ok"
        )

        litellm.completion(
            model="gpt-4",
            messages=[{"role": "user", "content": "test"}],
            temperature=0.5,
        )

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_acompletion_with_vertex_location(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that vertex_location is handled during async completion calls."""
        mock_acompletion = mocker.patch(
            f"{litellm_patch.__name__}._original_acompletion"
        )
        mock_acompletion.return_value = mock_judge_llm_response(
            prompt_tokens=10, completion_tokens=5, cache_hit=False, content="ok"
        )

        old_location = getattr(litellm, "vertex_location", None)

        await litellm.acompletion(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            vertex_location="europe-west1",
        )

        mock_acompletion.assert_called_once()
        call_kwargs = mock_acompletion.call_args[1]
        assert "vertex_location" not in call_kwargs
        assert getattr(litellm, "vertex_location", None) == old_location

    @pytest.mark.asyncio
    async def test_acompletion_with_both_vertex_params(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Test that both vertex params are handled during async completion."""
        mock_acompletion = mocker.patch(
            f"{litellm_patch.__name__}._original_acompletion"
        )
        mock_acompletion.return_value = mock_judge_llm_response(
            prompt_tokens=10, completion_tokens=5, cache_hit=False, content="ok"
        )

        old_location = getattr(litellm, "vertex_location", None)
        old_project = getattr(litellm, "vertex_project", None)

        await litellm.acompletion(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            vertex_location="us-central1",
            vertex_project="my-project",
        )

        call_kwargs = mock_acompletion.call_args[1]
        assert "vertex_location" not in call_kwargs
        assert "vertex_project" not in call_kwargs
        assert getattr(litellm, "vertex_location", None) == old_location
        assert getattr(litellm, "vertex_project", None) == old_project


class TestInterleavedSyncAsyncCompletions:
    """Test that sync and async completions sharing one lock don't race."""

    @pytest.mark.asyncio
    async def test_interleaved_sync_async_no_deadlock(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Interleaved sync/async vertex completions complete without deadlock."""
        response = mock_judge_llm_response(
            prompt_tokens=10, completion_tokens=5, cache_hit=False, content="ok"
        )
        mocker.patch.object(
            litellm_patch, "_original_completion", return_value=response
        )
        mocker.patch.object(
            litellm_patch, "_original_acompletion", return_value=response
        )

        old_vp = getattr(litellm, "vertex_project", None)
        old_vl = getattr(litellm, "vertex_location", None)
        completed: list[str] = []

        def run_sync(project: str, location: str) -> None:
            litellm.completion(
                model="vertex_ai/gemini-pro",
                messages=[{"role": "user", "content": "test"}],
                vertex_project=project,
                vertex_location=location,
            )
            completed.append(f"sync-{project}")

        async def run_async(project: str, location: str) -> None:
            await litellm.acompletion(
                model="vertex_ai/gemini-pro",
                messages=[{"role": "user", "content": "test"}],
                vertex_project=project,
                vertex_location=location,
            )
            completed.append(f"async-{project}")

        try:
            loop = asyncio.get_running_loop()
            await asyncio.wait_for(
                asyncio.gather(
                    loop.run_in_executor(None, run_sync, "proj-s1", "loc-s1"),
                    loop.run_in_executor(None, run_sync, "proj-s2", "loc-s2"),
                    run_async("proj-a1", "loc-a1"),
                    run_async("proj-a2", "loc-a2"),
                ),
                timeout=10,
            )

            assert len(completed) == 4
        finally:
            litellm.vertex_project = old_vp
            litellm.vertex_location = old_vl

    @pytest.mark.asyncio
    async def test_sequential_sync_async_restores_globals(
        self, mocker: MockerFixture, mock_judge_llm_response: Callable[..., Any]
    ) -> None:
        """Sequential sync then async vertex call restores globals correctly."""
        response = mock_judge_llm_response(
            prompt_tokens=10, completion_tokens=5, cache_hit=False, content="ok"
        )
        mocker.patch.object(
            litellm_patch, "_original_completion", return_value=response
        )
        mocker.patch.object(
            litellm_patch, "_original_acompletion", return_value=response
        )

        old_vp = getattr(litellm, "vertex_project", None)
        old_vl = getattr(litellm, "vertex_location", None)

        litellm.completion(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            vertex_project="proj-sync",
            vertex_location="loc-sync",
        )
        assert getattr(litellm, "vertex_project", None) == old_vp
        assert getattr(litellm, "vertex_location", None) == old_vl

        await litellm.acompletion(
            model="vertex_ai/gemini-pro",
            messages=[{"role": "user", "content": "test"}],
            vertex_project="proj-async",
            vertex_location="loc-async",
        )
        assert getattr(litellm, "vertex_project", None) == old_vp
        assert getattr(litellm, "vertex_location", None) == old_vl

    def test_sync_and_async_share_same_lock(self) -> None:
        """Both _vertex_override and _vertex_override_async use litellm_state_lock."""
        assert isinstance(litellm_patch.litellm_state_lock, type(threading.Lock()))
        assert not hasattr(litellm_patch, "litellm_state_async_lock")
