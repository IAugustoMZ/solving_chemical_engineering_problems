#!/bin/bash

set -e

echo "=========================================="
echo "Pre-Push Checks: Tests + Quality"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if poetry is available
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed${NC}"
    exit 1
fi

# Run tests
echo "Running tests..."
if poetry run pytest tests/ -v --tb=short; then
    echo -e "${GREEN}✓ All tests passed${NC}"
else
    echo -e "${RED}✗ Tests failed${NC}"
    exit 1
fi
echo ""

# Run quality checks
echo "Running quality checks..."
bash "$(dirname "$0")/quality-check.sh"

echo ""
echo -e "${GREEN}Ready to push!${NC}"
