"""Unit tests for SQL storage backend."""

# pylint: disable=redefined-outer-name,protected-access

import os
import tempfile
from typing import Generator

import pytest
from pytest_mock import MockerFixture
from sqlalchemy import create_engine, text

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.storage import (
    EvaluationResultDB,
    RunInfo,
    SQLStorageBackend,
    StorageError,
)


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def temp_db_url(temp_db_path: str) -> str:
    """Create a SQLAlchemy connection URL for the temp database."""
    return f"sqlite:///{temp_db_path}"


@pytest.fixture
def sample_result() -> EvaluationResult:
    """Create a sample evaluation result for testing."""
    return EvaluationResult(
        conversation_group_id="conv_001",
        tag="test",
        turn_id="turn_1",
        metric_identifier="ragas:faithfulness",
        metric_metadata='{"threshold": 0.8}',
        result="PASS",
        score=0.95,
        threshold=0.8,
        reason="Good response",
        query="What is Python?",
        response="Python is a programming language.",
        execution_time=1.5,
        api_input_tokens=100,
        api_output_tokens=50,
    )


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    """Create multiple sample evaluation results."""
    return [
        EvaluationResult(
            conversation_group_id="conv_001",
            metric_identifier="ragas:faithfulness",
            result="PASS",
            score=0.95,
        ),
        EvaluationResult(
            conversation_group_id="conv_001",
            metric_identifier="ragas:relevancy",
            result="PASS",
            score=0.88,
        ),
        EvaluationResult(
            conversation_group_id="conv_002",
            metric_identifier="nlp:bleu",
            result="FAIL",
            score=0.45,
        ),
    ]


class TestSQLStorageBackendProperties:
    """Tests for SQLStorageBackend properties."""

    def test_default_backend_name(self, temp_db_url: str) -> None:
        """Test default backend_name is 'database'."""
        backend = SQLStorageBackend(temp_db_url)
        assert backend.backend_name == "database"

    def test_custom_backend_name(self, temp_db_url: str) -> None:
        """Test custom backend_name."""
        backend = SQLStorageBackend(temp_db_url, backend_name="sqlite")
        assert backend.backend_name == "sqlite"

    def test_connection_url(self, temp_db_url: str) -> None:
        """Test connection_url property."""
        backend = SQLStorageBackend(temp_db_url)
        assert backend.connection_url == temp_db_url


class TestSQLStorageBackendInitialize:
    """Tests for initialize() method."""

    def test_creates_database_and_table(
        self, temp_db_path: str, temp_db_url: str
    ) -> None:
        """Test initialize() creates database file and table."""
        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo(name="test"))

        assert os.path.exists(temp_db_path)

        engine = create_engine(temp_db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='evaluation_results'"
                )
            )
            assert len(result.fetchall()) == 1
        backend.close()

    def test_resets_results_count(self, temp_db_url: str) -> None:
        """Test initialize() resets results count."""
        backend = SQLStorageBackend(temp_db_url)
        backend._results_count = 10

        backend.initialize(RunInfo())
        assert backend.results_count == 0
        backend.close()

    def test_initialize_fails_on_stale_table_schema(
        self, temp_db_url: str, mocker: MockerFixture
    ) -> None:
        """Existing evaluation_results table must define every ORM column."""
        engine = create_engine(temp_db_url)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE evaluation_results ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "run_id VARCHAR(36) NOT NULL"
                    ")"
                )
            )
        engine.dispose()

        backend = SQLStorageBackend(temp_db_url, backend_name="sqlite")
        real_engine = create_engine(temp_db_url, echo=False)
        dispose_spy = mocker.patch.object(
            real_engine, "dispose", wraps=real_engine.dispose
        )
        mocker.patch(
            "lightspeed_evaluation.core.storage.sql_storage.create_engine",
            return_value=real_engine,
        )

        with pytest.raises(StorageError, match="schema mismatch"):
            backend.initialize(RunInfo())

        dispose_spy.assert_called_once()
        assert backend._engine is None


class TestSQLStorageBackendSaveResult:
    """Tests for save_result() method."""

    def test_persists_data(
        self, temp_db_url: str, sample_result: EvaluationResult
    ) -> None:
        """Test save_result() persists data correctly."""
        backend = SQLStorageBackend(temp_db_url)
        run_info = RunInfo()
        backend.initialize(run_info)
        backend.save_result(sample_result)
        backend.close()

        engine = create_engine(temp_db_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM evaluation_results"))
            row = result.fetchone()

        assert row is not None
        assert row.run_id == run_info.run_id
        assert row.conversation_group_id == "conv_001"
        assert row.metric_identifier == "ragas:faithfulness"
        assert row.result == "PASS"
        assert row.score == 0.95
        assert backend.results_count == 1

    def test_without_initialize_raises(
        self, temp_db_url: str, sample_result: EvaluationResult
    ) -> None:
        """Test save_result() raises error if not initialized."""
        backend = SQLStorageBackend(temp_db_url)
        with pytest.raises(StorageError, match="not initialized"):
            backend.save_result(sample_result)

    def test_multiple_results(
        self, temp_db_url: str, sample_results: list[EvaluationResult]
    ) -> None:
        """Test saving multiple results individually."""
        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo())

        for result in sample_results:
            backend.save_result(result)
        backend.close()

        engine = create_engine(temp_db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) as cnt FROM evaluation_results")
            )
            assert result.fetchone().cnt == 3  # type: ignore[union-attr]


class TestSQLStorageBackendSaveRun:
    """Tests for save_run() method."""

    def test_batch_save(
        self, temp_db_url: str, sample_results: list[EvaluationResult]
    ) -> None:
        """Test save_run() saves all results in batch."""
        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo())
        backend.save_run(sample_results)

        assert backend.results_count == 3
        backend.close()

    def test_without_initialize_raises(
        self, temp_db_url: str, sample_results: list[EvaluationResult]
    ) -> None:
        """Test save_run() raises error if not initialized."""
        backend = SQLStorageBackend(temp_db_url)
        with pytest.raises(StorageError, match="not initialized"):
            backend.save_run(sample_results)

    def test_empty_list(self, temp_db_url: str) -> None:
        """Test save_run() with empty list."""
        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo())
        backend.save_run([])

        assert backend.results_count == 0
        backend.close()


class TestSQLStorageBackendClose:  # pylint: disable=too-few-public-methods
    """Tests for close() method."""

    def test_disposes_engine(self, temp_db_url: str) -> None:
        """Test close() disposes database engine."""
        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo())
        backend.close()

        assert backend._engine is None
        assert backend._session_factory is None


class TestSQLStorageBackendDataIntegrity:
    """Tests for data integrity and serialization."""

    def test_all_fields_stored(self, temp_db_url: str) -> None:
        """Test all EvaluationResult fields are stored."""
        result = EvaluationResult(
            conversation_group_id="conv_full",
            tag="integration",
            turn_id="turn_5",
            metric_identifier="custom:metric",
            metric_metadata='{"key": "value"}',
            result="PASS",
            score=0.92,
            threshold=0.75,
            reason="Excellent response",
            query="Complex question?",
            response="Detailed answer.",
            execution_time=2.5,
            api_input_tokens=200,
            api_output_tokens=150,
            judge_llm_input_tokens=50,
            judge_llm_output_tokens=25,
            time_to_first_token=0.1,
            streaming_duration=1.5,
            tokens_per_second=100.0,
            tool_calls='[{"name": "search"}]',
            contexts='["context1", "context2"]',
            expected_response="Expected answer",
            expected_intent="question",
            expected_keywords='[["keyword1"]]',
            expected_tool_calls='[{"name": "expected_tool"}]',
        )

        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo())
        backend.save_result(result)
        backend.close()

        engine = create_engine(temp_db_url)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM evaluation_results")).fetchone()

        assert row is not None
        assert row.conversation_group_id == "conv_full"
        assert row.score == 0.92
        assert row.execution_time == 2.5
        assert row.tool_calls == '[{"name": "search"}]'

    def test_null_fields_handled(self, temp_db_url: str) -> None:
        """Test null/optional fields are handled correctly."""
        result = EvaluationResult(
            conversation_group_id="conv_minimal",
            metric_identifier="simple:metric",
            result="PASS",
        )

        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo())
        backend.save_result(result)
        backend.close()

        engine = create_engine(temp_db_url)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM evaluation_results")).fetchone()

        assert row is not None
        assert row.score is None
        assert row.threshold is None

    def test_list_serialization(self, temp_db_url: str) -> None:
        """Test expected_response list is serialized to JSON."""
        result = EvaluationResult(
            conversation_group_id="conv_list",
            metric_identifier="test:metric",
            result="PASS",
            expected_response=["answer1", "answer2"],
        )

        backend = SQLStorageBackend(temp_db_url)
        backend.initialize(RunInfo())
        backend.save_result(result)
        backend.close()

        engine = create_engine(temp_db_url)
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT expected_response FROM evaluation_results")
            ).fetchone()

        assert row is not None
        assert row.expected_response == '["answer1", "answer2"]'


class TestSQLStorageBackendMultipleRuns:  # pylint: disable=too-few-public-methods
    """Tests for multiple evaluation runs."""

    def test_multiple_runs_same_database(
        self, temp_db_url: str, sample_result: EvaluationResult
    ) -> None:
        """Test multiple runs writing to same database."""
        run_info1 = RunInfo(name="run1")
        run_info2 = RunInfo(name="run2")

        backend1 = SQLStorageBackend(temp_db_url)
        backend1.initialize(run_info1)
        backend1.save_result(sample_result)
        backend1.close()

        backend2 = SQLStorageBackend(temp_db_url)
        backend2.initialize(run_info2)
        backend2.save_result(sample_result)
        backend2.close()

        engine = create_engine(temp_db_url)
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT DISTINCT run_id FROM evaluation_results")
            )
            run_ids = [row.run_id for row in result.fetchall()]

        assert len(run_ids) == 2
        assert run_info1.run_id in run_ids
        assert run_info2.run_id in run_ids


class TestEvaluationResultDB:
    """Tests for EvaluationResultDB SQLAlchemy model."""

    def test_table_name(self) -> None:
        """Test table name is correct."""
        assert EvaluationResultDB.__tablename__ == "evaluation_results"

    def test_all_csv_columns_present(self) -> None:
        """Test all CSV columns are in the DB model."""
        columns = {c.name for c in EvaluationResultDB.__table__.columns}
        required_columns = {
            "id",
            "run_id",
            "timestamp",
            "conversation_group_id",
            "tag",
            "turn_id",
            "metric_identifier",
            "metric_metadata",
            "result",
            "score",
            "threshold",
            "reason",
            "query",
            "response",
            "execution_time",
            "api_input_tokens",
            "api_output_tokens",
            "judge_llm_input_tokens",
            "judge_llm_output_tokens",
            "judge_scores",
            "time_to_first_token",
            "streaming_duration",
            "tokens_per_second",
            "tool_calls",
            "contexts",
            "expected_response",
            "expected_intent",
            "expected_keywords",
            "expected_tool_calls",
        }
        assert required_columns == columns
