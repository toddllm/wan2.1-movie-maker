#!/bin/bash
# Run enhanced video generation in two steps:
# 1. Enhance prompts using templates (if needed)
# 2. Generate videos with enhanced prompts

set -e  # Exit on error

# Default values
INPUT_FILE=""
SECONDS=10
URL="http://localhost:5001"
SKIP_ENHANCE=false
FORCE=false

# Path to the virtual environment
VENV_PATH="/home/tdeshane/development/wan-video/wan2.1/venv"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input=*)
            INPUT_FILE="${1#*=}"
            shift
            ;;
        --seconds=*)
            SECONDS="${1#*=}"
            shift
            ;;
        --url=*)
            URL="${1#*=}"
            shift
            ;;
        --skip-enhance)
            SKIP_ENHANCE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --input=PROMPT_FILE [--seconds=SECONDS] [--url=URL] [--skip-enhance] [--force]"
            exit 1
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required."
    echo "Usage: $0 --input=PROMPT_FILE [--seconds=SECONDS] [--url=URL] [--skip-enhance] [--force]"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# Check if GPU is already in use
if command -v nvidia-smi &> /dev/null; then
    # Check if a generation process is already running
    if ps -ef | grep -q "[g]enerate.py"; then
        echo "Warning: Another video generation process is already running."
        echo "Options:"
        echo "  1. Wait for the current process to complete"
        echo "  2. Run with --force to proceed anyway (may cause errors)"
        echo "  3. Kill the current process with: kill \$(ps -ef | grep 'generate.py' | grep -v grep | awk '{print \$2}')"
        
        if [ "$FORCE" = false ]; then
            exit 1
        else
            echo "Proceeding anyway as --force was specified."
        fi
    else
        # Check for very high GPU memory usage (>5GB) without a generate.py process
        GPU_USAGE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{print $1}')
        if [ "$GPU_USAGE" -gt 5000 ] && [ "$FORCE" = false ]; then
            echo "Warning: GPU has high memory usage (${GPU_USAGE}MB) but no generate.py process was found."
            echo "Another application might be using the GPU."
            echo "Run with --force to proceed anyway."
            exit 1
        fi
    fi
fi

# Check if the input file is already in the enhanced_prompts directory
if [[ "$INPUT_FILE" == *"enhanced_prompts"* && "$INPUT_FILE" == *"_only.txt" ]]; then
    echo "Using already enhanced prompts file: $INPUT_FILE"
    ENHANCED_FILE="$INPUT_FILE"
    SKIP_ENHANCE=true
else
    # Create timestamp for output files
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    ENHANCED_DIR="enhanced_prompts"
    mkdir -p "$ENHANCED_DIR"
    ENHANCED_FILE="${ENHANCED_DIR}/enhanced_${TIMESTAMP}_only.txt"
    
    if [ "$SKIP_ENHANCE" = false ]; then
        echo "Step 1: Enhancing prompts from $INPUT_FILE"
        echo "Using template-based enhancement (no model required)"

        # Run the prompt enhancement script
        python enhance_prompts.py "$INPUT_FILE" --output="${ENHANCED_DIR}/enhanced_${TIMESTAMP}.txt"

        # Check if the enhanced file exists
        if [ ! -f "$ENHANCED_FILE" ]; then
            echo "Error: Enhanced prompts file not created. Check for errors above."
            exit 1
        fi
    else
        echo "Skipping enhancement step as requested."
        # Just copy the input file to the enhanced file location
        cp "$INPUT_FILE" "$ENHANCED_FILE"
    fi
fi

echo ""
echo "Step 2: Generating videos with prompts"
echo "Using video length: $SECONDS seconds"
echo "Using server URL: $URL"
echo "Using prompts from: $ENHANCED_FILE"

# Activate the virtual environment for video generation
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Run the batch generation script with the enhanced prompts
echo "Running video generation..."
python batch_generate.py "$ENHANCED_FILE" --seconds="$SECONDS" --url="$URL"

# Deactivate the virtual environment
deactivate

echo ""
echo "Process complete!"
if [ "$SKIP_ENHANCE" = false ]; then
    echo "Enhanced prompts saved to: ${ENHANCED_DIR}/enhanced_${TIMESTAMP}.txt"
fi
echo "Videos generated from prompts in: $ENHANCED_FILE" 