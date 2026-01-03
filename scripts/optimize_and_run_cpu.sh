#!/bin/bash
# Script to first quantize a model and then run generation on CPU.

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Find the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Set the base directory to the project root (one level up from 'scripts')
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# --- Functions ---
show_usage() {
    echo "Usage: $0 <model_name> [generate_options...]"
    echo ""
    echo "This script first quantizes the specified model for CPU optimization,"
    echo "and then runs the generation script with the quantized model."
    echo ""
    echo "Arguments:"
    echo "  <model_name>      : The name of the model directory inside 'models/' to optimize and run."
    echo "  [generate_options...]: Optional arguments to pass to the generate.py script."
    echo ""
    echo "Example:"
    echo "  $0 my_cool_model --prompt \"Hello world\" --max_new_tokens 100"
}

# --- Argument Parsing ---
if [ -z "$1" ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    show_usage
    exit 0
fi

MODEL_NAME=$1
shift # Remove the model name from arguments, the rest are for generate.py

MODEL_PATH="$BASE_DIR/models/$MODEL_NAME"
QUANTIZED_MODEL_PATH="${MODEL_PATH}_quantized"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at '$MODEL_PATH'"
    exit 1
fi

# --- Main Logic ---

# Activate the virtual environment if it exists
if [ -d "$BASE_DIR/venv" ]; then
  source "$BASE_DIR/venv/bin/activate"
  echo "Activated Python virtual environment."
fi

# 1. Run Quantization
echo "--------------------------------------------------"
echo "Step 1: Optimizing model '$MODEL_NAME' for CPU..."
echo "--------------------------------------------------"
python3 "$SCRIPT_DIR/quantize_cpu.py" --model_path "$MODEL_PATH"

# Check if quantization was successful
if [ ! -d "$QUANTIZED_MODEL_PATH" ]; then
    echo "Error: Quantization failed. The quantized model directory was not created."
    exit 1
fi
echo "Model quantized successfully. Optimized model saved to '$QUANTIZED_MODEL_PATH'"
echo ""

# 2. Run Generation with the Quantized Model
echo "--------------------------------------------------"
echo "Step 2: Running generation with optimized model..."
echo "--------------------------------------------------"
python3 "$SCRIPT_DIR/generate.py" --model_path "$QUANTIZED_MODEL_PATH" --hardware-strategy cpu --quantized "$@"

echo "--------------------------------------------------"
echo "Script finished."
echo "--------------------------------------------------"
