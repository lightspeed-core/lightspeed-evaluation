"""Test runner script for LSC Evaluation Framework tests."""

import os
import sys
from pathlib import Path
from typing import List, Optional

import pytest


def run_tests(
    test_path: Optional[str] = None,
    markers: Optional[List[str]] = None,
    verbose: bool = True,
    coverage: bool = False,
    parallel: bool = False,
    fail_fast: bool = False,
) -> int:
    """
    Run tests with specified options.
    
    Args:
        test_path: Specific test file or directory to run
        markers: List of pytest markers to include/exclude
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        parallel: Run tests in parallel (requires pytest-xdist)
        fail_fast: Stop on first failure
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Build pytest arguments
    args = []
    
    # Test path
    if test_path:
        args.append(test_path)
    else:
        # Always use tests/ directory relative to project root
        args.append("tests/")
    
    # Verbosity
    if verbose:
        args.extend(["-v", "--tb=short"])
    
    # Markers
    if markers:
        for marker in markers:
            if marker.startswith("not "):
                args.extend(["-m", marker])
            else:
                args.extend(["-m", marker])
    
    # Coverage
    if coverage:
        try:
            import pytest_cov
            args.extend([
                "--cov=lsc_eval",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-report=xml"
            ])
        except ImportError:
            print("⚠️  pytest-cov not installed. Install with: pip install pytest-cov")
            print("Running tests without coverage...")
            coverage = False
    
    # Parallel execution
    if parallel:
        args.extend(["-n", "auto"])
    
    # Fail fast
    if fail_fast:
        args.append("-x")
    
    # Additional options
    args.extend([
        "--strict-config",
        "--color=yes",
        "--durations=10"
    ])
    
    print(f"Running pytest with args: {' '.join(args)}")
    return pytest.main(args)


def run_unit_tests() -> int:
    """Run only unit tests."""
    return run_tests(markers=["unit"])


def run_integration_tests() -> int:
    """Run only integration tests."""
    return run_tests(markers=["integration"])


def run_config_tests() -> int:
    """Run configuration-related tests."""
    return run_tests(markers=["config"])


def run_model_tests() -> int:
    """Run Pydantic model tests."""
    return run_tests(markers=["models"])


def run_validation_tests() -> int:
    """Run data validation tests."""
    return run_tests(markers=["validation"])


def run_output_tests() -> int:
    """Run output generation tests."""
    return run_tests(markers=["output"])


def run_fast_tests() -> int:
    """Run fast tests (exclude slow tests)."""
    return run_tests(markers=["not slow"])


def run_all_tests() -> int:
    """Run all tests."""
    return run_tests()


def run_with_coverage() -> int:
    """Run all tests with coverage reporting."""
    try:
        import pytest_cov
        return run_tests(coverage=True)
    except ImportError:
        print("⚠️  pytest-cov not installed for coverage reporting.")
        print("Install with: pip install pytest-cov")
        print("Running tests without coverage instead...")
        return run_tests()


def main():
    """Main test runner entry point."""
    if len(sys.argv) < 2:
        print("LSC Evaluation Framework Test Runner")
        print("\nUsage:")
        print("  python test_runner.py <command> [options]")
        print("\nCommands:")
        print("  all          - Run all tests")
        print("  unit         - Run unit tests only")
        print("  integration  - Run integration tests only")
        print("  config       - Run configuration tests")
        print("  models       - Run Pydantic model tests")
        print("  validation   - Run data validation tests")
        print("  fast         - Run fast tests (exclude slow)")
        print("  output       - Run output generation tests")
        print("  coverage     - Run all tests with coverage")
        print("  file <path>  - Run specific test file")
        print("\nExamples:")
        print("  python test_runner.py all")
        print("  python test_runner.py unit")
        print("  python test_runner.py file tests/core/test_models.py")
        print("  python test_runner.py coverage")
        return 1
    
    command = sys.argv[1].lower()
    
    # Set up environment
    project_root = Path(__file__).parent.parent  # Go up from tests/ to lsc_eval/
    test_dir = Path(__file__).parent  # tests/ directory
    
    # Change to the project root (lsc_eval/)
    os.chdir(project_root)
    
    # Add src to Python path
    src_path = project_root / "src"
    if src_path.exists():
        sys.path.insert(0, str(src_path))
    
    # Run appropriate test command
    if command == "all":
        return run_all_tests()
    elif command == "unit":
        return run_unit_tests()
    elif command == "integration":
        return run_integration_tests()
    elif command == "config":
        return run_config_tests()
    elif command == "models":
        return run_model_tests()
    elif command == "validation":
        return run_validation_tests()
    elif command == "output":
        return run_output_tests()
    elif command == "fast":
        return run_fast_tests()
    elif command == "coverage":
        return run_with_coverage()
    elif command == "file" and len(sys.argv) > 2:
        test_file = sys.argv[2]
        return run_tests(test_path=test_file)
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
