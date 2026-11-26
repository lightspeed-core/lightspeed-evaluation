r"""Dynamic Dataset Curator - CLI entry point.

Post-processes an existing evaluation run into curated dataset splits. It does
not run evaluation; point it at the results summary JSON and the evaluation-data
YAML that was evaluated.

    dataset-curator \
        --results output/evaluation_<ts>_summary.json \
        --eval-data pcm_eval_data.yaml \
        --output-dir output/curate
"""

import argparse
import logging
import sys

from lightspeed_evaluation.core.curate import BUCKETS, ScopeOptions, curate

logger = logging.getLogger(__name__)


def create_curate_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the curate runner."""
    parser = argparse.ArgumentParser(
        description="Dynamic Dataset Curator - split an eval run into datasets",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the evaluation results summary JSON (*_summary.json)",
    )
    parser.add_argument(
        "--eval-data",
        default=None,
        help=(
            "Optional evaluation-data YAML for richer turn content and as a signals "
            "fallback; omit it when the results already carry signals"
        ),
    )
    parser.add_argument(
        "--curate-config",
        default=None,
        help="Optional curate.yaml holding overrides",
    )
    parser.add_argument(
        "--output-dir",
        default="output/curate",
        help="Directory for split YAMLs and report (default: output/curate)",
    )
    parser.add_argument(
        "--conversation-scope",
        action="store_true",
        help=(
            "Propagate each conversation's signals to all its turns (label the "
            "whole arc, not just the signal-bearing turn)"
        ),
    )
    parser.add_argument(
        "--drop-signal-turn",
        action="store_true",
        help=(
            "With --conversation-scope, exclude the signal-bearing turns from the "
            "output instead of classifying them on their own signal"
        ),
    )
    return parser


def run_curate(args: argparse.Namespace) -> bool:
    """Run curation; return True on success."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = curate(
            args.results,
            args.eval_data,
            args.output_dir,
            args.curate_config,
            ScopeOptions(args.conversation_scope, args.drop_signal_turn),
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"\n❌ Curate failed: {e}")
        return False

    print("\n🎉 Curate complete!")
    for bucket in BUCKETS:
        print(f"   {bucket:>12}: {report['bucket_turn_counts'].get(bucket, 0)} turns")
    print(f"   {'skipped':>12}: {report['skipped_turns']} turns (no eval result)")
    print(f"   {'review items':>12}: {len(report['needs_review'])}")
    print(f"\n   Splits + report written to: {args.output_dir}")
    return True


def main() -> int:
    """Command line interface."""
    args = create_curate_parser().parse_args()
    return 0 if run_curate(args) else 1


if __name__ == "__main__":
    sys.exit(main())
