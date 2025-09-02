#!/usr/bin/env python3
"""Test runner script for LightSpeed Evaluation Framework."""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, markers=None):
    """Run tests with specified options."""
    
    # Use the same Python executable that's running this script
    python_exe = sys.executable
    
    # Base pytest command
    cmd = [python_exe, "-m", "pytest"]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage if requested
    if coverage:
        cmd.extend([
            "--cov=lightspeed_evaluation",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml"
        ])
    
    # Add markers if specified
    if markers:
        cmd.extend(["-m", markers])
    
    # Get the tests directory (current directory since we're inside tests/)
    tests_dir = Path(__file__).parent
    
    # Add test selection based on type
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "config":
        cmd.extend(["-m", "config"])
    elif test_type == "metrics":
        cmd.extend(["-m", "metrics"])
    elif test_type == "cli":
        cmd.extend(["-m", "cli"])
    elif test_type == "output":
        cmd.extend(["-m", "output"])
    elif test_type == "slow":
        cmd.extend(["-m", "slow"])
    elif test_type == "fast":
        cmd.extend(["-m", "not slow"])
    elif test_type != "all":
        # Specific test file or pattern
        # If it's a relative path, make it relative to tests directory
        if not test_type.startswith("/") and not test_type.startswith("tests/"):
            test_type = str(tests_dir / test_type)
        cmd.append(test_type)
    
    # Add tests directory for general test types
    if test_type == "all" or test_type in ["unit", "integration", "config", "metrics", "cli", "output", "slow", "fast"]:
        cmd.append(str(tests_dir))
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main function for test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for LightSpeed Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  {sys.executable} tests/run_tests.py                    # Run all tests
  {sys.executable} tests/run_tests.py --type unit        # Run only unit tests
  {sys.executable} tests/run_tests.py --type integration # Run only integration tests
  {sys.executable} tests/run_tests.py --type config      # Run only config tests
  {sys.executable} tests/run_tests.py --type metrics     # Run only metrics tests
  {sys.executable} tests/run_tests.py --type cli         # Run only CLI tests
  {sys.executable} tests/run_tests.py --type fast        # Run fast tests (exclude slow)
  {sys.executable} tests/run_tests.py --coverage         # Run with coverage report
  {sys.executable} tests/run_tests.py --verbose          # Run with verbose output
  {sys.executable} tests/run_tests.py --markers "unit and not slow"  # Custom markers
  {sys.executable} tests/run_tests.py test_config.py     # Run specific test file
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["all", "unit", "integration", "config", "metrics", "cli", "output", "slow", "fast"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests with verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run tests with coverage report"
    )
    
    parser.add_argument(
        "--markers", "-m",
        help="Custom pytest markers (e.g., 'unit and not slow')"
    )
    
    parser.add_argument(
        "test_path",
        nargs="?",
        help="Specific test file or directory to run"
    )
    
    args = parser.parse_args()
    
    # Use test_path if provided, otherwise use type
    test_type = args.test_path if args.test_path else args.type
    
    # Use the same Python executable that's running this script
    python_exe = sys.executable
    
    # Check if pytest is available
    try:
        subprocess.run([python_exe, "-m", "pytest", "--version"], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Error: pytest is not installed. Please install it with:")
        print(f"  {python_exe} -m pip install pytest")
        if args.coverage:
            print(f"  {python_exe} -m pip install pytest-cov  # for coverage support")
        return 1
    
    # Check if coverage is requested but not available
    if args.coverage:
        try:
            subprocess.run([python_exe, "-m", "pytest_cov", "--version"], 
                          check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Warning: pytest-cov is not installed. Coverage disabled.")
            print(f"Install it with: {python_exe} -m pip install pytest-cov")
            args.coverage = False
    
    # Run the tests
    return run_tests(
        test_type=test_type,
        verbose=args.verbose,
        coverage=args.coverage,
        markers=args.markers
    )


if __name__ == "__main__":
    sys.exit(main())
