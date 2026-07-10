"""Offline validation for the agentic proposal scenario library.

These tests need no cluster: they load every scenario data file with the
same system config the CLI runner uses, so schema drift, broken script
paths, or metrics missing from ``metrics_metadata`` fail fast in CI
instead of mid-eval on a live cluster.
"""

from pathlib import Path

import pytest

from lightspeed_evaluation import ConfigLoader
from lightspeed_evaluation.core.system import DataValidator

INTEGRATION_TEST_DIR = Path(__file__).parent
SYSTEM_CONFIG_PATH = INTEGRATION_TEST_DIR / "system-config-agents-proposal.yaml"
SCENARIO_DATA_FILES = [
    INTEGRATION_TEST_DIR / "test_evaluation_data_proposal.yaml",
    *sorted((INTEGRATION_TEST_DIR / "agentic" / "scenarios").glob("*.yaml")),
]


@pytest.mark.parametrize("data_file", SCENARIO_DATA_FILES, ids=lambda p: p.name)
def test_scenario_data_validates_with_runner_config(data_file: Path) -> None:
    """Validate each scenario data file exactly as the CLI runner would.

    Passing ``system_config`` activates metric-name validation against
    ``metrics_metadata`` — the code path the cluster-bound proposal tests
    do not exercise.
    """
    loader = ConfigLoader()
    system_config = loader.load_system_config(str(SYSTEM_CONFIG_PATH))
    validator = DataValidator(
        api_enabled=True,
        fail_on_invalid_data=True,
        system_config=system_config,
    )
    data = validator.load_evaluation_data(str(data_file))
    assert data, f"{data_file.name} contains no conversations"
