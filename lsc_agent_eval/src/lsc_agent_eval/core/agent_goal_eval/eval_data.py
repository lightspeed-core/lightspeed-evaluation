"""Agent Goal Eval data management."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from ..utils.exceptions import EvaluationDataError
from .models import ConversationDataConfig

logger = logging.getLogger(__name__)


class AgentGoalEvalDataManager:
    """Processes agent eval data and validation."""

    def __init__(self, eval_data_file: str) -> None:
        """Initialize eval data manager."""
        self.eval_data_file = eval_data_file
        self.conversations: list[ConversationDataConfig] = []

        self._load_eval_data()
        self._log_loaded_data_stats()

    def _load_eval_data(self) -> None:
        """Load evaluation data from YAML file."""
        try:
            eval_data_path = Path(self.eval_data_file).resolve()
            logger.info("Loading evaluation data from: %s", str(eval_data_path))

            with open(eval_data_path, "r", encoding="utf-8") as file:
                raw_data = yaml.safe_load(file)

            if raw_data is None:
                raise EvaluationDataError("Eval data file is empty")
            if not isinstance(raw_data, list):
                raise EvaluationDataError(
                    f"Eval data file must contain a list of conversations, got {type(raw_data)}"
                )
            if not raw_data:
                raise EvaluationDataError("Eval data file must contain at least one conversation")

            logger.info("Found %d conversation(s) in YAML file", len(raw_data))

            # Process each conversation
            self._load_conversation_data(raw_data)

        except yaml.YAMLError as e:
            raise EvaluationDataError(f"Invalid YAML in eval data file: {e}") from e
        except FileNotFoundError as e:
            raise EvaluationDataError(f"Eval data file not found: {e}") from e
        except EvaluationDataError:
            raise
        except Exception as e:
            raise EvaluationDataError(f"Error loading eval data file: {e}") from e

    def _load_conversation_data(self, raw_data: list[dict[str, Any]]) -> None:
        """Load conversation data."""
        logger.info("Processing conversation data...")

        self.conversations = []
        processed_groups = set()

        for idx, conversation_data in enumerate(raw_data, 1):
            logger.debug("Processing conversation %d", idx)

            try:
                conversation_config = ConversationDataConfig(**conversation_data)

                # Check for duplicate conversation groups
                if conversation_config.conversation_group in processed_groups:
                    raise EvaluationDataError(
                        "Duplicate conversation_group "
                        f"'{conversation_config.conversation_group}' found"
                    )
                processed_groups.add(conversation_config.conversation_group)

                # Store the conversation
                self.conversations.append(conversation_config)

                logger.info(
                    "Loaded conversation '%s' with %d evaluations",
                    conversation_config.conversation_group,
                    len(conversation_config.conversation),
                )

            except ValidationError as e:
                error_details = self._format_pydantic_error(e)
                conversation_group = conversation_data.get(
                    "conversation_group", f"conversation_{idx}"
                )
                raise EvaluationDataError(
                    f"Validation error in conversation '{conversation_group}': {error_details}"
                ) from e
            except EvaluationDataError:
                raise
            except Exception as e:
                raise EvaluationDataError(f"Error processing conversation {idx}: {e}") from e

    def _format_pydantic_error(self, error: ValidationError) -> str:
        """Format Pydantic validation error."""
        errors = []
        for err in error.errors():
            field = " -> ".join(str(loc) for loc in err["loc"])
            message = err["msg"]
            errors.append(f"{field}: {message}")
        return "; ".join(errors)

    def _log_loaded_data_stats(self) -> None:
        """Log statistics about loaded data."""
        if not self.conversations:
            raise EvaluationDataError("No valid conversations found in eval data file")

        # Calculate statistics from conversations
        eval_types: dict[str, int] = {}
        conversation_stats = {}
        total_evaluations = 0

        for conversation in self.conversations:
            conv_group = conversation.conversation_group
            conversation_stats[conv_group] = len(conversation.conversation)
            total_evaluations += len(conversation.conversation)

            for eval_config in conversation.conversation:
                for eval_type in eval_config.eval_types:
                    eval_types[eval_type] = eval_types.get(eval_type, 0) + 1

        if total_evaluations == 0:
            raise EvaluationDataError("No valid evaluations found in eval data file")

        # Check for duplicate eval_ids across all conversations
        all_eval_ids = []
        for conversation in self.conversations:
            all_eval_ids.extend([eval_config.eval_id for eval_config in conversation.conversation])

        duplicate_ids = [eval_id for eval_id in all_eval_ids if all_eval_ids.count(eval_id) > 1]
        if duplicate_ids:
            logger.warning(
                "Duplicate eval_id(s) found across conversations: %s",
                set(duplicate_ids),
            )

        logger.info("✅ Data validation complete:")
        logger.info("  %d conversations", len(self.conversations))
        logger.info("  %d total evaluations", total_evaluations)
        logger.info("  Evaluation types: %s", dict(eval_types))

        for conv_group, count in conversation_stats.items():
            logger.debug("  %s: %d evaluations", conv_group, count)

    def get_conversations(self) -> list[ConversationDataConfig]:
        """Get all conversation configurations."""
        return self.conversations

    def get_eval_count(self) -> int:
        """Get the total number of evaluation configurations."""
        return sum(len(conversation.conversation) for conversation in self.conversations)
