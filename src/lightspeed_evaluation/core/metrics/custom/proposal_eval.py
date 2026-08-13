"""Deprecated: Use agentic_run_eval instead.

This module is deprecated and maintained for backward compatibility only.
Use lightspeed_evaluation.core.metrics.custom.agentic_run_eval instead.
"""

import logging

# Re-export from agentic_run_eval for backward compatibility
from lightspeed_evaluation.core.metrics.custom.agentic_run_eval import (
    evaluate_agentic_run_status as evaluate_proposal_status,
)

__all__ = ["evaluate_proposal_status"]

logger = logging.getLogger(__name__)
logger.warning(
    "Module 'lightspeed_evaluation.core.metrics.custom.proposal_eval' is deprecated. "
    "Use 'lightspeed_evaluation.core.metrics.custom.agentic_run_eval' instead."
)
