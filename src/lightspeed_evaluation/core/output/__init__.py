"""Output handling - Reports and visualization.

Uses lazy imports to defer loading heavy visualization libraries
(matplotlib, seaborn) until they're actually needed.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightspeed_evaluation.core.output.data_persistence import save_evaluation_data
    from lightspeed_evaluation.core.output.generator import OutputHandler
    from lightspeed_evaluation.core.output.visualization import GraphGenerator

__all__ = ["OutputHandler", "GraphGenerator", "save_evaluation_data"]


def __getattr__(name: str) -> Any:
    """Lazy import mechanism for output module.

    GraphGenerator imports matplotlib and seaborn, which are heavy.
    These are only loaded when visualization is actually requested.
    """
    # pylint: disable=import-outside-toplevel
    if name == "OutputHandler":
        from lightspeed_evaluation.core.output.generator import OutputHandler

        return OutputHandler
    if name == "GraphGenerator":
        from lightspeed_evaluation.core.output.visualization import GraphGenerator

        return GraphGenerator
    if name == "save_evaluation_data":
        from lightspeed_evaluation.core.output.data_persistence import (
            save_evaluation_data,
        )

        return save_evaluation_data

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
