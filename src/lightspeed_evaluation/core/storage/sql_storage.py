"""SQL storage backend for evaluation results.

This module provides a SQLite storage backend using SQLAlchemy that persists
evaluation results.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from lightspeed_evaluation.core.models import EvaluationResult
from lightspeed_evaluation.core.system.exceptions import StorageError
from lightspeed_evaluation.core.storage.protocol import BaseStorageBackend, RunInfo

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):  # pylint: disable=too-few-public-methods
    """Base class for SQLAlchemy ORM models."""


class EvaluationResultDB(Base):  # pylint: disable=too-few-public-methods
    """SQLAlchemy model for evaluation results.

    This table mirrors the CSV output columns plus run metadata
    (run_id and timestamp) for tracking evaluation runs.
    """

    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    conversation_group_id = Column(String(255), nullable=False, index=True)
    tag = Column(String(100), nullable=True)
    turn_id = Column(String(100), nullable=True)
    metric_identifier = Column(String(255), nullable=False, index=True)
    metric_metadata = Column(Text, nullable=True)
    result = Column(String(20), nullable=False, index=True)
    score = Column(Float, nullable=True)
    threshold = Column(Float, nullable=True)
    reason = Column(Text, nullable=True)
    query = Column(Text, nullable=True)
    response = Column(Text, nullable=True)
    execution_time = Column(Float, nullable=True)
    api_input_tokens = Column(Integer, nullable=True)
    api_output_tokens = Column(Integer, nullable=True)
    judge_llm_input_tokens = Column(Integer, nullable=True)
    judge_llm_output_tokens = Column(Integer, nullable=True)
    judge_scores = Column(Text, nullable=True)
    time_to_first_token = Column(Float, nullable=True)
    streaming_duration = Column(Float, nullable=True)
    tokens_per_second = Column(Float, nullable=True)
    tool_calls = Column(Text, nullable=True)
    contexts = Column(Text, nullable=True)
    expected_response = Column(Text, nullable=True)
    expected_intent = Column(Text, nullable=True)
    expected_keywords = Column(Text, nullable=True)
    expected_tool_calls = Column(Text, nullable=True)


class SQLStorageBackend(BaseStorageBackend):
    """Database storage backend implementation using SQLAlchemy.

    This backend persists evaluation results to a SQLite database using SQLAlchemy.

    Attributes:
        connection_url: SQLAlchemy connection URL.
        table_name: Name of the table to store results.
    """

    def __init__(
        self,
        connection_url: str,
        table_name: str = "evaluation_results",
        backend_name: str = "database",
    ):
        """Initialize database storage backend.

        Args:
            connection_url: SQLAlchemy connection URL (e.g., "sqlite:///./results.db").
            table_name: Name of the table (reserved for future dynamic table support).
            backend_name: Name identifier for this backend instance.
        """
        self._connection_url = connection_url
        self._table_name = table_name
        self._backend_name = backend_name
        self._engine: Any = None
        self._session_factory: Optional[
            sessionmaker[Session]  # pylint: disable=unsubscriptable-object
        ] = None
        self._run_info: Optional[RunInfo] = None
        self._results_count = 0

    @property
    def backend_name(self) -> str:
        """Return the name of this storage backend."""
        return self._backend_name

    @property
    def connection_url(self) -> str:
        """Return the database connection URL."""
        return self._connection_url

    @property
    def results_count(self) -> int:
        """Return the number of results saved in this run."""
        return self._results_count

    def initialize(self, run_info: RunInfo) -> None:
        """Initialize the database backend for a new evaluation run.

        Creates the tables if they don't exist.

        Args:
            run_info: Information about the evaluation run.

        Raises:
            StorageError: If database initialization fails.
        """
        self._run_info = run_info
        self._results_count = 0

        try:
            self._engine = create_engine(self._connection_url, echo=False)
            Base.metadata.create_all(self._engine)
            self._session_factory = sessionmaker(bind=self._engine)
            logger.info(
                "Database backend initialized: %s (run_id=%s)",
                self._backend_name,
                run_info.run_id,
            )
        except SQLAlchemyError as e:
            raise StorageError(
                f"Failed to initialize database: {e}",
                backend_name=self.backend_name,
            ) from e

    def save_result(self, result: EvaluationResult) -> None:
        """Save a single evaluation result to the database.

        Args:
            result: The evaluation result to save.

        Raises:
            StorageError: If saving fails.
        """
        if self._session_factory is None or self._run_info is None:
            raise StorageError(
                "Backend not initialized. Call initialize() first.",
                backend_name=self.backend_name,
            )

        try:
            db_record = self._result_to_db_record(result)
            with self._session_factory() as session:
                session.add(db_record)
                session.commit()
            self._results_count += 1
        except SQLAlchemyError as e:
            raise StorageError(
                f"Failed to save result to database: {e}",
                backend_name=self.backend_name,
            ) from e

    def save_run(self, results: list[EvaluationResult]) -> None:
        """Save all evaluation results in batch.

        Args:
            results: List of all evaluation results.

        Raises:
            StorageError: If saving fails.
        """
        if self._session_factory is None or self._run_info is None:
            raise StorageError(
                "Backend not initialized. Call initialize() first.",
                backend_name=self.backend_name,
            )

        try:
            db_records = [self._result_to_db_record(r) for r in results]
            with self._session_factory() as session:
                session.add_all(db_records)
                session.commit()
            self._results_count += len(results)
            logger.info(
                "Batch saved %d results to database (run_id=%s)",
                len(results),
                self._run_info.run_id,
            )
        except SQLAlchemyError as e:
            raise StorageError(
                f"Failed to batch save results to database: {e}",
                backend_name=self.backend_name,
            ) from e

    def finalize(self) -> None:
        """Finalize the storage backend.

        Raises:
            StorageError: If finalization fails.
        """
        if self._run_info is None:
            return

        try:
            logger.info(
                "Database backend finalized: %d results saved (run_id=%s)",
                self._results_count,
                self._run_info.run_id,
            )
        except Exception as e:
            raise StorageError(
                f"Failed to finalize database backend: {e}",
                backend_name=self.backend_name,
            ) from e

    def close(self) -> None:
        """Close the database connection and release resources."""
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
        self._session_factory = None
        self._run_info = None
        logger.debug("Database backend closed: %s", self._backend_name)

    def _result_to_db_record(self, result: EvaluationResult) -> EvaluationResultDB:
        """Convert an EvaluationResult to a database record.

        Args:
            result: The evaluation result to convert.

        Returns:
            A database record ready for insertion.
        """
        if self._run_info is None:
            raise StorageError(
                "Backend not initialized. Call initialize() first.",
                backend_name=self.backend_name,
            )

        return EvaluationResultDB(
            run_id=self._run_info.run_id,
            timestamp=datetime.now(timezone.utc),
            conversation_group_id=result.conversation_group_id,
            tag=result.tag,
            turn_id=result.turn_id,
            metric_identifier=result.metric_identifier,
            metric_metadata=result.metric_metadata,
            result=result.result,
            score=result.score,
            threshold=result.threshold,
            reason=result.reason,
            query=result.query,
            response=result.response,
            execution_time=result.execution_time,
            api_input_tokens=result.api_input_tokens,
            api_output_tokens=result.api_output_tokens,
            judge_llm_input_tokens=result.judge_llm_input_tokens,
            judge_llm_output_tokens=result.judge_llm_output_tokens,
            judge_scores=self._serialize_judge_scores(result.judge_scores),
            time_to_first_token=result.time_to_first_token,
            streaming_duration=result.streaming_duration,
            tokens_per_second=result.tokens_per_second,
            tool_calls=result.tool_calls,
            contexts=result.contexts,
            expected_response=self._serialize_expected_response(
                result.expected_response
            ),
            expected_intent=result.expected_intent,
            expected_keywords=result.expected_keywords,
            expected_tool_calls=result.expected_tool_calls,
        )

    @staticmethod
    def _serialize_judge_scores(judge_scores: Any) -> Optional[str]:
        """Serialize judge scores to JSON string.

        Args:
            judge_scores: Judge scores data (list of JudgeScore objects or None).

        Returns:
            JSON string or None if no scores.
        """
        if judge_scores is None:
            return None
        try:
            scores_data = [
                {"judge_id": s.judge_id, "score": s.score, "reason": s.reason}
                for s in judge_scores
            ]
            return json.dumps(scores_data)
        except (TypeError, AttributeError):
            logger.warning(
                "Failed to serialize judge_scores: unexpected structure %r",
                type(judge_scores).__name__,
            )
            return None

    @staticmethod
    def _serialize_expected_response(
        expected_response: Optional[str | list[str]],
    ) -> Optional[str]:
        """Serialize expected response to string.

        Args:
            expected_response: Single string or list of strings.

        Returns:
            JSON string for lists, or the original string.
        """
        if expected_response is None:
            return None
        if isinstance(expected_response, list):
            return json.dumps(expected_response)
        return expected_response
