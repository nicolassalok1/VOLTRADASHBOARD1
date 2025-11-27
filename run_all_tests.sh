#!/usr/bin/env bash
# Run the full test suite under ./tests and stream logs to the terminal.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "=== Running tests via unittest discover (tests/) verbose..."
python3 -m unittest discover -v -s tests -p "test*.py" -t . "$@" | tee tests/testsGPT/test_run.log
