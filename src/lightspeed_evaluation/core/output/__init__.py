"""Output handling - Reports and visualization."""

from typing import TYPE_CHECKING

from lightspeed_evaluation.core.system.lazy_import import create_lazy_getattr

if TYPE_CHECKING:
    # ruff: noqa: F401
    from lightspeed_evaluation.core.output.data_persistence import (
        save_evaluation_data,
    )
    from lightspeed_evaluation.core.output.generator import OutputHandler
    from lightspeed_evaluation.core.output.visualization import GraphGenerator

_LAZY_IMPORTS = {
    "OutputHandler": (
        "lightspeed_evaluation.core.output.generator",
        "OutputHandler",
    ),
    "GraphGenerator": (
        "lightspeed_evaluation.core.output.visualization",
        "GraphGenerator",
    ),
    "save_evaluation_data": (
        "lightspeed_evaluation.core.output.data_persistence",
        "save_evaluation_data",
    ),
}

__getattr__ = create_lazy_getattr(_LAZY_IMPORTS, __name__)
