#!/bin/bash
# Test runner script for G-code fingerprinting project

set -e  # Exit on error

echo "========================================="
echo "G-Code Fingerprinting Test Suite"
echo "========================================="
echo ""

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    fi
fi

# Parse command line arguments
COVERAGE=true
VERBOSE=false
MARKERS=""
PARALLEL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cov)
            COVERAGE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -m|--markers)
            MARKERS="-m $2"
            shift 2
            ;;
        -n|--parallel)
            PARALLEL=true
            shift
            ;;
        -h|--help)
            echo "Usage: ./scripts/run_tests.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --no-cov          Disable coverage reporting"
            echo "  -v, --verbose     Verbose output"
            echo "  -m MARKERS        Run only tests with specific markers (e.g., 'unit', 'not slow')"
            echo "  -n, --parallel    Run tests in parallel"
            echo "  -h, --help        Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./scripts/run_tests.sh                    # Run all tests with coverage"
            echo "  ./scripts/run_tests.sh --no-cov          # Run without coverage"
            echo "  ./scripts/run_tests.sh -m 'not slow'     # Run only fast tests"
            echo "  ./scripts/run_tests.sh -n                # Run tests in parallel"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build pytest command
CMD="pytest tests/"

if [ "$VERBOSE" = true ]; then
    CMD="$CMD -vv"
fi

if [ "$COVERAGE" = true ]; then
    CMD="$CMD --cov=src/miracle --cov-report=html --cov-report=term-missing"
else
    CMD="$CMD --no-cov"
fi

if [ -n "$MARKERS" ]; then
    CMD="$CMD $MARKERS"
fi

if [ "$PARALLEL" = true ]; then
    CMD="$CMD -n auto"
fi

# Run tests
echo "Running: $CMD"
echo ""
$CMD

# Display coverage report location if coverage was generated
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "========================================="
    echo "Coverage report generated at: htmlcov/index.html"
    echo "========================================="
fi
