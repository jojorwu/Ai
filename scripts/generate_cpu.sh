#!/bin/bash
# Script to run generation on CPU

# Source the common setup script
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Run the generation script with CPU hardware strategy
python3 "$SCRIPT_DIR/generate.py" --hardware-strategy cpu "$@"
