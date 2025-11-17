"""API client module for actual data generation."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.core.api.client import APIClient

__all__ = ["APIClient"]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for API module."""
    # pylint: disable=import-outside-toplevel
    if name == "APIClient":
        from lightspeed_evaluation.core.api.client import APIClient

        return APIClient

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
