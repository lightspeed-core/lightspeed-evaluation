"""Evaluation Pipeline - Main evaluation orchestrator."""

import logging
from typing import Optional

from ...core.api import APIClient
from ...core.models import EvaluationData, EvaluationResult
from ...core.output.data_persistence import save_evaluation_data
from ...core.system import ConfigLoader, DataValidator
from .amender import APIDataAmender
from .errors import EvaluationErrorHandler
from .evaluator import MetricsEvaluator
from .processor import ConversationProcessor

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Evaluation pipeline - orchestrates the evaluation process through different stages.

    Responsibilities:
    - Initialize and coordinate components
    - Validate data
    - Orchestrate evaluation flow
    - Collect results
    - Save updated data
    """

    def __init__(self, config_loader: ConfigLoader, output_dir: Optional[str] = None):
        """Initialize evaluation pipeline with config and create components."""
        self.config_loader = config_loader
        if not config_loader.system_config:
            raise ValueError("SystemConfig must be loaded before initializing pipeline")
        self.original_data_path: Optional[str] = None
        self.output_dir = output_dir or config_loader.system_config.output.output_dir

        # Initialize components
        self._initialize_components()
        logger.info("Evaluation Pipeline initialized")

    def _initialize_components(self) -> None:
        """Initialize all required components."""
        # Data validator
        config = self.config_loader.system_config
        if config is None:
            raise ValueError(
                "SystemConfig must be loaded before initializing components"
            )
        self.data_validator = DataValidator(api_enabled=config.api.enabled)

        # Create pipeline components
        api_client = self._create_api_client()
        api_amender = APIDataAmender(api_client)
        error_handler = EvaluationErrorHandler()
        metrics_evaluator = MetricsEvaluator(self.config_loader)
        # Group components for easier access
        self.components = {
            "api_client": api_client,
            "api_amender": api_amender,
            "error_handler": error_handler,
            "metrics_evaluator": metrics_evaluator,
        }

        # Conversation processor
        self.conversation_processor = ConversationProcessor(
            self.config_loader,
            self.components["metrics_evaluator"],
            self.components["api_amender"],
            self.components["error_handler"],
        )

    def _create_api_client(self) -> Optional[APIClient]:
        """Create API client if enabled."""
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded before creating API client")
        if not config.api.enabled:
            return None

        api_config = config.api
        logger.info("Setting up API client: %s", api_config.api_base)

        client = APIClient(
            api_base=api_config.api_base,
            config={
                "provider": api_config.provider,
                "model": api_config.model,
                "no_tools": api_config.no_tools,
                "system_prompt": api_config.system_prompt,
            },
            endpoint_type=api_config.endpoint_type,
            timeout=api_config.timeout,
        )

        logger.info("API client initialized for %s endpoint", api_config.endpoint_type)
        return client

    def validate_data(self, evaluation_data: list[EvaluationData]) -> bool:
        """Validate evaluation data using data validator."""
        return self.data_validator.validate_evaluation_data(evaluation_data)

    def run_evaluation(
        self,
        evaluation_data: list[EvaluationData],
        original_data_path: Optional[str] = None,
    ) -> list[EvaluationResult]:
        """Run evaluation on provided data.

        Args:
            evaluation_data: List of conversation data to evaluate
            original_data_path: Path to original data file for saving updates

        Returns:
            List of evaluation results.
        """
        self.original_data_path = original_data_path
        logger.info("Starting evaluation")
        results: list[EvaluationResult] = []

        # Step 1: Validate data
        logger.info("Validating data")
        if not self.validate_data(evaluation_data):
            raise ValueError("Data validation failed. Cannot proceed with evaluation.")

        # Step 2: Process each conversation
        logger.info("Processing conversations")
        for conv_data in evaluation_data:
            conv_results = self.conversation_processor.process_conversation(conv_data)
            results.extend(conv_results)

        # Step 3: Save updated data if API was used
        config = self.config_loader.system_config
        if config is None:
            raise ValueError("SystemConfig must be loaded")
        if config.api.enabled:
            logger.info("Saving updated evaluation data")
            self._save_updated_data(evaluation_data)

        logger.info("Evaluation complete: %d results generated", len(results))
        return results

    def _save_updated_data(self, evaluation_data: list[EvaluationData]) -> None:
        """Save updated evaluation data with API amendments."""
        if not self.original_data_path:
            logger.warning("No original data path available, cannot save updated data")
            return

        try:
            updated_file = save_evaluation_data(
                evaluation_data, self.original_data_path, self.output_dir
            )
            if updated_file:
                logger.info("Updated data saved: %s", updated_file)
                logger.info(
                    "To use amended data without new API calls, "
                    "disable the API call using system config"
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Don't fail the evaluation if saving fails
            logger.warning("Failed to save updated data: %s", e)

    def close(self) -> None:
        """Clean up resources."""
        api_client = self.components.get("api_client")
        if api_client:
            api_client.close()
