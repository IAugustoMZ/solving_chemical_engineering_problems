#!/bin/bash

set -e

echo "=========================================="
echo "Running Code Quality Checks"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed${NC}"
    echo "Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Check if virtual environment is active
if [[ -z "$VIRTUAL_ENV" ]] && ! poetry env info -p &> /dev/null; then
    echo -e "${YELLOW}Warning: Virtual environment not detected${NC}"
    echo "Installing dependencies with poetry..."
    poetry install
    echo ""
fi

# 1. Black - Code formatting check
echo "1. Checking code formatting with black..."
if poetry run black --check src/ tests/ 2>/dev/null; then
    echo -e "${GREEN}✓ Black check passed${NC}"
else
    echo -e "${RED}✗ Black check failed${NC}"
    echo "  Run: poetry run black src/ tests/"
    FAILED=$((FAILED + 1))
fi
echo ""

# 2. isort - Import sorting check
echo "2. Checking import sorting with isort..."
if poetry run isort --check-only src/ tests/ 2>/dev/null; then
    echo -e "${GREEN}✓ isort check passed${NC}"
else
    echo -e "${RED}✗ isort check failed${NC}"
    echo "  Run: poetry run isort src/ tests/"
    FAILED=$((FAILED + 1))
fi
echo ""

# 3. flake8 - Linting check
echo "3. Running flake8 linter..."
if poetry run flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics 2>/dev/null; then
    echo -e "${GREEN}✓ flake8 check passed${NC}"
else
    echo -e "${YELLOW}⚠ flake8 found issues (non-critical)${NC}"
fi
echo ""

# 4. pylint - Advanced linting
echo "4. Running pylint..."
if poetry run pylint src/ --disable=all --enable=E,F 2>/dev/null; then
    echo -e "${GREEN}✓ pylint check passed${NC}"
else
    echo -e "${YELLOW}⚠ pylint found issues (non-critical)${NC}"
fi
echo ""

# 5. pylint duplicate code check
echo "5. Checking for duplicate code..."
if poetry run pylint src/ --disable=all --enable=duplicate-code 2>/dev/null; then
    echo -e "${GREEN}✓ No duplicate code detected${NC}"
else
    echo -e "${YELLOW}⚠ Potential duplicate code found (review carefully)${NC}"
fi
echo ""

# Summary
echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All critical quality checks passed!${NC}"
    exit 0
else
    echo -e "${RED}$FAILED critical check(s) failed${NC}"
    echo ""
    echo "Quick fix commands:"
    echo "  poetry run black src/ tests/"
    echo "  poetry run isort src/ tests/"
    exit 1
fi
