#!/usr/bin/env bash
# Run the minepick unit tests. No network, no API keys needed.
#
# Usage:
#   ./run_tests.sh            # create/reuse .venv, install pytest, run tests
#   ./run_tests.sh -q         # pass extra args through to pytest
set -euo pipefail
cd "$(dirname "$0")"

PY=python3

# Reuse an existing venv if present; create one if pytest isn't importable anywhere.
if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import pytest' 2>/dev/null; then
    PY=.venv/bin/python
else
    if $PY -c 'import pytest' 2>/dev/null; then
        : # system python already has pytest
    else
        echo "==> pytest not found; creating .venv and installing pytest..."
        $PY -m venv .venv
        ./.venv/bin/pip install --quiet pytest
        PY=.venv/bin/python
    fi
fi

echo "==> Running tests with $PY"
exec $PY -m pytest tests/ -v "$@"
