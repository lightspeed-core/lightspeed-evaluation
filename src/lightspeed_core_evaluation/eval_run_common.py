"""Common code for evaluation runner scripts."""

import argparse


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to the parser."""
    parser.add_argument(
        "--judge_provider",
        default="watsonx",
        type=str,
        help="Provider name for judge model; required for LLM based evaluation",
    )
    parser.add_argument(
        "--judge_model",
        default="meta-llama/llama-3-1-8b-instruct",
        type=str,
        help="Judge model; required for LLM based evaluation",
    )
    parser.add_argument(
        "--eval_out_dir",
        default=None,
        type=str,
        help="Result destination.",
    )
