r"""Trace ingestion - CLI entry point.

Reads observations (from the Langfuse API or an export file, per the config)
and writes an evaluation-data YAML whose turns carry ``signals``, ready for
``lightspeed-eval`` and then ``dataset-curator``.

    trace-ingest \
        --config config/ingest.yaml \
        --output pcm_eval_data.yaml

For an ``api`` source, the host comes from the ``LANGFUSE_HOST`` env var and
credentials from ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``. A ``file``
source needs no credentials.
"""

import argparse
import logging
import sys

import httpx

from lightspeed_evaluation.core.ingestion import ingest

logger = logging.getLogger(__name__)


def create_ingest_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the ingest runner."""
    parser = argparse.ArgumentParser(
        description="Ingest traces into evaluation data with signals",
    )
    parser.add_argument(
        "--config",
        default="config/ingest.yaml",
        help="Path to the ingestion config YAML (default: config/ingest.yaml)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the evaluation-data YAML",
    )
    return parser


def run_ingest(args: argparse.Namespace) -> bool:
    """Run ingestion; return True on success."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        summary = ingest(args.config, args.output)
    except (FileNotFoundError, ValueError, httpx.HTTPError) as e:
        print(f"\n❌ Ingest failed: {e}")
        return False

    print("\n🎉 Ingest complete!")
    print(f"   observations:  {summary['observations']}")
    print(f"   conversations: {summary['conversations']}")
    print(f"   turns:         {summary['turns']}")
    print(f"\n   Eval data written to: {summary['output_path']}")
    return True


def main() -> int:
    """Command line interface."""
    args = create_ingest_parser().parse_args()
    return 0 if run_ingest(args) else 1


if __name__ == "__main__":
    sys.exit(main())
