"""CLI interface for comparing evaluation runs."""

import argparse
import json
import sys
import traceback
from pathlib import Path

from lightspeed_evaluation.core.output.comparison import EvaluationComparison


def main() -> int:
    """Main CLI entry point for evaluation comparison."""
    args = _parse_arguments()

    # Validate arguments
    validation_error = _validate_arguments(args)
    if validation_error:
        return validation_error

    try:
        return _run_comparison(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in summary file: {e}", file=sys.stderr)
        return 1
    except (ValueError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


def _parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare two evaluation runs and determine statistical significance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two evaluation summary files
  python3 -m lightspeed_evaluation.cli.compare run1_summary.json run2_summary.json
  
  # Compare with custom significance level
  python3 -m lightspeed_evaluation.cli.compare run1_summary.json run2_summary.json --alpha 0.01
  
  # Save comparison results to JSON file
  python3 -m lightspeed_evaluation.cli.compare run1_summary.json run2_summary.json --output comparison_results.json
  
  # Generate only the report (no JSON output)
  python3 -m lightspeed_evaluation.cli.compare run1_summary.json run2_summary.json --report-only
        """,
    )

    parser.add_argument(
        "summary1", type=str, help="Path to first evaluation summary JSON file"
    )
    parser.add_argument(
        "summary2", type=str, help="Path to second evaluation summary JSON file"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Path to save comparison results as JSON file"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate and display the report, don't save JSON results",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output except for errors"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def _validate_arguments(args: argparse.Namespace) -> int:
    """Validate command line arguments."""
    if args.alpha <= 0 or args.alpha >= 1:
        print("Error: Alpha must be between 0 and 1", file=sys.stderr)
        return 1

    if not Path(args.summary1).exists():
        print(f"Error: Summary file not found: {args.summary1}", file=sys.stderr)
        return 1

    if not Path(args.summary2).exists():
        print(f"Error: Summary file not found: {args.summary2}", file=sys.stderr)
        return 1

    return 0


def _run_comparison(args: argparse.Namespace) -> int:
    """Run the evaluation comparison."""
    comparison = EvaluationComparison(alpha=args.alpha)

    if args.verbose:
        _print_verbose_info(args)

    if not args.quiet:
        print("Performing statistical comparison...")

    comparison_results = comparison.compare_evaluations(args.summary1, args.summary2)

    if not args.quiet:
        report = comparison.generate_report(comparison_results)
        print(report)

    if args.output and not args.report_only:
        _save_results_to_file(args.output, comparison_results, args.quiet)

    return _determine_exit_code(comparison_results, args.quiet)


def _print_verbose_info(args: argparse.Namespace) -> None:
    """Print verbose information about the comparison."""
    print("Comparing evaluation runs:")
    print(f"  Run 1: {args.summary1}")
    print(f"  Run 2: {args.summary2}")
    print(f"  Significance level: α = {args.alpha}")
    print()


def _save_results_to_file(
    output_path: str, comparison_results: dict, quiet: bool
) -> None:
    """Save comparison results to a JSON file."""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)

    if not quiet:
        print(f"\nComparison results saved to: {output_path_obj}")


def _determine_exit_code(comparison_results: dict, quiet: bool) -> int:
    """Determine the exit code based on comparison results."""
    significant_metrics = [
        metric
        for metric, comparison_data in comparison_results.get(
            "metric_comparisons", {}
        ).items()
        if comparison_data.get("statistical_significance", {}).get(
            "overall_significant", False
        )
    ]

    if significant_metrics:
        if not quiet:
            print(
                f"\nFound statistically significant differences in "
                f"{len(significant_metrics)} metrics."
            )
        return 0  # Success, but with significant differences

    if not quiet:
        print("\nNo statistically significant differences found.")
    return 0  # Success, no significant differences


if __name__ == "__main__":
    sys.exit(main())
