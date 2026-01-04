#!/bin/bash
# Script to run generation on GPU

# Find the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the base directory to the project root (one level up from 'scripts')
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment if it exists
if [ -d "$BASE_DIR/venv" ]; then
  source "$BASE_DIR/venv/bin/activate"
fi

# Run the generation script with discrete hardware strategy for GPU
# For a potential speedup on supported hardware, uncomment the --torch_compile flag.
# Note: This may increase initial startup time.
python3 "$SCRIPT_DIR/generate.py" --hardware-strategy discrete "$@" # --torch_compile
