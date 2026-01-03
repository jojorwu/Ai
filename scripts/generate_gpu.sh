#!/bin/bash
# Script to run generation on GPU

# Source the common setup script
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Run the generation script with discrete hardware strategy for GPU
# For a potential speedup on supported hardware, uncomment the --torch_compile flag.
# Note: This may increase initial startup time.
python3 "$SCRIPT_DIR/generate.py" --hardware-strategy discrete "$@" # --torch_compile
