#!/bin/bash
# This script contains common setup logic for other shell scripts in this directory.
# It should be sourced, not executed directly.

# Exit immediately if a command exits with a non-zero status.
set -e

# SCRIPT_DIR is the absolute path to the 'scripts' directory.
# BASH_SOURCE[0] is the path to this script (_common.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# BASE_DIR is the project root, which is one level up from the scripts directory.
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Activate the Python virtual environment if it exists in the project root.
if [ -d "$BASE_DIR/venv" ]; then
  source "$BASE_DIR/venv/bin/activate"
fi
