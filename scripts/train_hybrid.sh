#!/bin/bash
# Script to run training in hybrid mode (CPU + GPU)

# Source the common setup script
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Run the training script with hybrid hardware strategy
# For a potential speedup on supported hardware, uncomment the --torch_compile flag.
# Note: This may increase initial startup time.
python3 "$SCRIPT_DIR/train.py" --hardware-strategy hybrid "$@" # --torch_compile
