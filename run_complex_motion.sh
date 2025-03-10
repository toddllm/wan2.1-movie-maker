#!/bin/bash
# Run batch generation of complex motion prompts

set -e  # Exit on error

# Default values
SECONDS=10
URL="http://localhost:5001"
ENHANCED_FILE="enhanced_prompts/enhanced_20250310_120656_only.txt"

echo "Starting batch generation of complex motion videos"
echo "Using enhanced prompts from: $ENHANCED_FILE"
echo "Video length: $SECONDS seconds"
echo "Server URL: $URL"
echo ""
echo "This will generate 9 videos with complex motion, which will take approximately 3 hours."
echo ""

# Check if the enhanced file exists
if [ ! -f "$ENHANCED_FILE" ]; then
    echo "Error: Enhanced prompts file not found: $ENHANCED_FILE"
    echo "Please run ./enhance_only.sh --input=complex_motion_prompts.txt first."
    exit 1
fi

# Path to the virtual environment
VENV_PATH="/home/tdeshane/development/wan-video/wan2.1/venv"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Run the batch generation script
echo "Starting batch generation..."
python batch_generate.py "$ENHANCED_FILE" --seconds="$SECONDS" --url="$URL" --delay=30

# Deactivate the virtual environment
deactivate

echo ""
echo "Batch generation complete!"
echo "Check the clips directory for the generated videos." 