"""API client module for actual data generation."""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from lightspeed_evaluation.core.api.client import APIClient

_LAZY_IMPORTS = {
    "APIClient": ("lightspeed_evaluation.core.api.client", "APIClient"),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
