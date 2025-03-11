#!/bin/bash

# run_quality_check.sh - Script to run video quality evaluation for batch processing
# Usage: ./run_quality_check.sh [input_file] [min_score]

# Default values
INPUT_FILE=${1:-"dynamic_prompts.txt"}
MIN_SCORE=${2:-7}
API_URL="http://localhost:5001"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="enhanced_prompts/quality_check_${TIMESTAMP}.json"
LOG_FILE="quality_check_${TIMESTAMP}.log"

echo "Starting video quality evaluation at $(date)"
echo "Input file: $INPUT_FILE"
echo "Minimum score: $MIN_SCORE"
echo "API URL: $API_URL"
echo "Output file: $OUTPUT_FILE"
echo "Log file: $LOG_FILE"

# Create enhanced_prompts directory if it doesn't exist
mkdir -p enhanced_prompts

# Run the evaluation script
python evaluate_videos.py "$INPUT_FILE" --min-score "$MIN_SCORE" --api-url "$API_URL" --output "$OUTPUT_FILE" 2>&1 | tee "$LOG_FILE"

# Extract prompts that need regeneration
echo "Extracting prompts that need regeneration..."
REGENERATE_FILE="enhanced_prompts/regenerate_${TIMESTAMP}.txt"
python -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)
with open('$REGENERATE_FILE', 'w') as f:
    for item in data:
        if item.get('enhanced_prompt') and not item.get('new_video'):
            f.write(item['enhanced_prompt'] + '\n')
"

# Count how many prompts need regeneration
REGENERATE_COUNT=$(wc -l < "$REGENERATE_FILE")
echo "Found $REGENERATE_COUNT prompts that need regeneration"

if [ "$REGENERATE_COUNT" -gt 0 ]; then
    echo "Running batch generation for prompts that need regeneration..."
    python batch_generate.py "$REGENERATE_FILE" --seconds 10
fi

echo "Video quality evaluation completed at $(date)"
echo "Results saved to $OUTPUT_FILE"
echo "Log saved to $LOG_FILE"

# Print summary
echo "Summary:"
echo "  Total prompts processed: $(wc -l < "$INPUT_FILE")"
echo "  Prompts that need regeneration: $REGENERATE_COUNT" 