#!/bin/bash
# Enhance prompts only (no video generation)
# This allows checking the enhanced prompts before generating videos

set -e  # Exit on error

# Default values
INPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input=*)
            INPUT_FILE="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input=PROMPT_FILE"
            exit 1
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required."
    echo "Usage: $0 --input=PROMPT_FILE"
    exit 1
fi

# Create timestamp for output files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ENHANCED_DIR="enhanced_prompts"
mkdir -p "$ENHANCED_DIR"

echo "Enhancing prompts from $INPUT_FILE"
echo "Using template-based enhancement"

# Run the prompt enhancement script
ENHANCED_FILE="${ENHANCED_DIR}/enhanced_${TIMESTAMP}_only.txt"
python enhance_prompts.py "$INPUT_FILE" --output="${ENHANCED_DIR}/enhanced_${TIMESTAMP}.txt"

# Check if the enhanced file exists
if [ ! -f "$ENHANCED_FILE" ]; then
    echo "Error: Enhanced prompts file not created. Check for errors above."
    exit 1
fi

echo ""
echo "Prompt enhancement complete!"
echo "Enhanced prompts saved to: ${ENHANCED_DIR}/enhanced_${TIMESTAMP}.txt"
echo "Enhanced prompts only file: $ENHANCED_FILE"
echo ""
echo "To view the enhanced prompts, run:"
echo "cat $ENHANCED_FILE"
echo ""
echo "To generate videos with these prompts, run:"
echo "./run_enhanced_generation.sh --input=$ENHANCED_FILE --seconds=1" 