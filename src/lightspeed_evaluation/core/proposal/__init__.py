"""Deprecated: Use core.agentic_run instead.

This module is deprecated and maintained for backward compatibility only.
Use lightspeed_evaluation.core.agentic_run instead.
"""

import logging

# Re-export from agentic_run for backward compatibility
from lightspeed_evaluation.core.agentic_run import derive_phase

__all__ = ["derive_phase"]

logger = logging.getLogger(__name__)
logger.warning(
    "Module 'lightspeed_evaluation.core.proposal' is deprecated. "
    "Use 'lightspeed_evaluation.core.agentic_run' instead."
)
