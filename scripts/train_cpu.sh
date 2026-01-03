#!/bin/bash
# Script to run training on CPU

# Source the common setup script
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Run the training script with CPU hardware strategy
python3 "$SCRIPT_DIR/train.py" --hardware-strategy cpu "$@"
