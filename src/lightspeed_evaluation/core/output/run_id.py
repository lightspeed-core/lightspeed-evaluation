"""This run's shared identifying timestamp.

Every file a run produces (reports and, when ``uaem_lib`` is configured,
its raw-score sink) is named ``..._<run id>...``. Pinned into the
environment on first use so every caller in the process shares one id,
regardless of call order.
"""

import os
from datetime import UTC, datetime

_RUN_ID_ENV = "LSE_RUN_ID"


def resolve_run_id() -> str:
    """Resolve this process's run id, pinning a fresh timestamp once.

    Returns:
        A ``%Y%m%d_%H%M%S`` timestamp, stable for the lifetime of the
        process once first resolved.
    """
    existing = os.environ.get(_RUN_ID_ENV)
    if existing:
        return existing

    run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    os.environ[_RUN_ID_ENV] = run_id
    return run_id
