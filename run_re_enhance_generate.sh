#!/bin/bash

# run_re_enhance_generate.sh - Script to re-enhance prompts and generate videos
# Usage: ./run_re_enhance_generate.sh [count] [seconds]

# Default values
COUNT=${1:-8}
SECONDS=${2:-10}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RE_ENHANCED_FILE="enhanced_prompts/re_enhanced_${TIMESTAMP}.txt"

echo "Starting re-enhancement and video generation at $(date)"
echo "Number of prompts to re-enhance: $COUNT"
echo "Video duration (seconds): $SECONDS"

# Create enhanced_prompts directory if it doesn't exist
mkdir -p enhanced_prompts

# Re-enhance prompts
echo "Re-enhancing prompts..."
./re_enhance_prompts.py --count $COUNT --output "$RE_ENHANCED_FILE"

# Generate videos from re-enhanced prompts
echo "Generating videos from re-enhanced prompts..."
./batch_generate.py "$RE_ENHANCED_FILE" --seconds $SECONDS

echo "Re-enhancement and video generation completed at $(date)"
echo "Re-enhanced prompts saved to: $RE_ENHANCED_FILE"
echo "Check the clips directory for generated videos" 