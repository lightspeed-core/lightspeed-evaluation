"""Utility functions for agent evaluation."""

import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to argument parser."""
    # Add common evaluation arguments
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
