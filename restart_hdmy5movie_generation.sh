#!/bin/bash

# HDMY 5 Movie Generation Restart Script
# This script restarts the video generation process from where it left off

# Set up environment
MOVIE_DIR="/home/tdeshane/movie_maker"
PROMPTS_FILE="$MOVIE_DIR/hdmy5movie_prompts_fixed_all.txt"
OUTPUT_DIR="$MOVIE_DIR/hdmy5movie_videos"
LOG_DIR="$MOVIE_DIR/logs"
START_INDEX=29  # Start from the 30th prompt (0-based index)

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR/02_prologue"
mkdir -p "$OUTPUT_DIR/03_act1"
mkdir -p "$MOVIE_DIR/hdmy5movie_posters/02_prologue"
mkdir -p "$MOVIE_DIR/hdmy5movie_posters/03_act1"

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_DIR/generation.log"
}

# Check if the prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    log_message "Error: Prompts file not found at $PROMPTS_FILE"
    exit 1
fi

# Count the number of prompts
PROMPT_COUNT=$(grep -c "." "$PROMPTS_FILE")
REMAINING_COUNT=$((PROMPT_COUNT - START_INDEX))
log_message "Found $PROMPT_COUNT total prompts in $PROMPTS_FILE"
log_message "Starting from index $START_INDEX ($REMAINING_COUNT prompts remaining)"

# Start the video generation
log_message "Starting video generation from where it left off..."
log_message "Output will be saved to $OUTPUT_DIR"

# Get the next prompt to be processed
NEXT_PROMPT=$(sed -n "$((START_INDEX+1))p" "$PROMPTS_FILE" | cut -c 1-80)
log_message "Next prompt to be processed: \"$NEXT_PROMPT...\""

# Run the Python script for video generation
# Starting from the specified index
nohup python3 "$MOVIE_DIR/generate_hdmy5movie.py" \
    --prompts "$PROMPTS_FILE" \
    --output "$OUTPUT_DIR" \
    --start "$START_INDEX" \
    >> "$LOG_DIR/generation.log" 2>&1 &

GENERATOR_PID=$!
log_message "Video generation restarted with PID $GENERATOR_PID"

# Save the PID to a file for later reference
echo "$GENERATOR_PID" > "$LOG_DIR/generator.pid"

log_message "Generation process restarted successfully!"
echo "Generation process restarted successfully!"
echo "To stop the process, run: kill \$(cat $LOG_DIR/generator.pid)"

# Provide instructions for monitoring
echo ""
echo "To monitor the generation progress:"
echo "  tail -f $LOG_DIR/generation.log" 