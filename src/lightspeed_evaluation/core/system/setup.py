"""System setup and initialization utilities."""

import logging
import os
from typing import Any, Dict

from ..models import LoggingConfig


def setup_environment_variables(config_data: Dict[str, Any]) -> None:
    """Setup environment variables from validated config data."""
    try:
        # Set environment variables from config
        env_config = config_data.get("environment", {})
        for env_var, value in env_config.items():
            os.environ[env_var] = str(value)

    except (KeyError, TypeError) as e:
        # Fallback to hardcoded defaults if config processing fails
        print(f"Warning: Could not process environment config: {e}")
        print("Using fallback environment settings...")
        os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
        os.environ["DEEPEVAL_DISABLE_PROGRESS_BAR"] = "YES"
        os.environ["LITELLM_LOG_LEVEL"] = "ERROR"


def setup_logging(logging_config: LoggingConfig) -> logging.Logger:
    """Configure logging for application and packages."""
    # Get logging settings from validated Pydantic model
    source_level = getattr(
        logging,
        logging_config.source_level.upper(),
        logging.INFO,
    )
    package_level = getattr(
        logging,
        logging_config.package_level.upper(),
        logging.WARNING,
    )
    log_format = logging_config.log_format

    # Configure root logger for our application
    logging.basicConfig(
        level=source_level,
        format=log_format,
        force=True,  # Override any existing configuration
    )

    # Set logging levels for packages using system configuration
    package_overrides = logging_config.package_overrides

    # Apply default package level to commonly problematic packages if not explicitly configured
    default_noisy_packages = [
        "urllib3",
        "requests",
        "httpx",
        "openai",
        "anthropic",
        "litellm",
        "ragas",
        "deepeval",
        "matplotlib",
        "seaborn",
    ]

    for package_name in default_noisy_packages:
        if package_name not in package_overrides:
            logging.getLogger(package_name).setLevel(package_level)

    # Apply specific package overrides from system configuration
    for package_name, override_level in package_overrides.items():
        try:
            override_level_obj = getattr(logging, override_level.upper())
            logging.getLogger(package_name).setLevel(override_level_obj)
        except AttributeError:
            print(
                f"Warning: Invalid log level '{override_level}' for package '{package_name}'"
            )

    # Get logger for our application
    logger = logging.getLogger("lightspeed_evaluation")
    logger.setLevel(source_level)

    logger.debug(
        "Logging configured: Source=%s, Packages=%s",
        logging.getLevelName(source_level),
        logging.getLevelName(package_level),
    )
    if package_overrides:
        logger.debug("Package overrides applied: %s", package_overrides)

    return logger
