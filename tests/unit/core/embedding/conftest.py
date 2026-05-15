"""Fixtures for embedding unit tests."""

import pytest
from pytest_mock import MockerFixture, MockType


@pytest.fixture
def mock_embedding_factory(mocker: MockerFixture) -> MockType:
    """Mock embedding_factory for ragas embedding tests."""
    return mocker.patch("lightspeed_evaluation.core.embedding.ragas.embedding_factory")
