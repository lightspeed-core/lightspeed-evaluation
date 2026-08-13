"""Deprecated: Use agentic_run_amender instead.

This module is deprecated and maintained for backward compatibility only.
Use lightspeed_evaluation.pipeline.evaluation.agentic_run_amender instead.
"""

import logging

from lightspeed_evaluation.pipeline.evaluation.agentic_run_amender import (
    STEP_RESOURCES,
)

# Re-export from agentic_run_amender for backward compatibility
from lightspeed_evaluation.pipeline.evaluation.agentic_run_amender import (
    AgenticRunAmender as ProposalAmender,
)

__all__ = ["ProposalAmender", "STEP_RESOURCES"]

logger = logging.getLogger(__name__)
logger.warning(
    "Module 'lightspeed_evaluation.pipeline.evaluation.proposal_amender' is deprecated. "
    "Use 'lightspeed_evaluation.pipeline.evaluation.agentic_run_amender' instead."
)
