#!/bin/bash

# HDMY 5 Movie Generation Starter Script
# This script starts both the video generation process and the HTML updater

# Set up environment
MOVIE_DIR="/home/tdeshane/movie_maker"
PROMPTS_FILE="$MOVIE_DIR/hdmy5movie_prompts.txt"
OUTPUT_DIR="$MOVIE_DIR/hdmy5movie_videos"
HTML_FILE="$MOVIE_DIR/hdmy5movie.html"
LOG_DIR="$MOVIE_DIR/logs"

# Create necessary directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Check if the prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file not found at $PROMPTS_FILE"
    exit 1
fi

# Count the number of prompts
PROMPT_COUNT=$(grep -c "." "$PROMPTS_FILE")
echo "Found $PROMPT_COUNT prompts in $PROMPTS_FILE"

# Check if Node.js is installed (for the updater)
if ! command -v node &> /dev/null; then
    echo "Warning: Node.js is not installed. The HTML updater will not run."
    RUN_UPDATER=false
else
    RUN_UPDATER=true
fi

# Start the HTML updater in the background
if [ "$RUN_UPDATER" = true ]; then
    echo "Starting HTML updater..."
    nohup node "$MOVIE_DIR/hdmy5movie_updater.js" > "$LOG_DIR/updater.log" 2>&1 &
    UPDATER_PID=$!
    echo "HTML updater started with PID $UPDATER_PID"
fi

# Start the video generation
echo "Starting video generation..."
echo "Output will be saved to $OUTPUT_DIR"
echo "Logs will be saved to $LOG_DIR/generation.log"

# Run the Python script for video generation
nohup python3 "$MOVIE_DIR/generate_hdmy5movie.py" \
    --prompts "$PROMPTS_FILE" \
    --output "$OUTPUT_DIR" \
    --model "/path/to/your/video/model" \
    > "$LOG_DIR/generation.log" 2>&1 &

GENERATOR_PID=$!
echo "Video generation started with PID $GENERATOR_PID"

# Save the PIDs to a file for later reference
echo "$GENERATOR_PID" > "$LOG_DIR/generator.pid"
if [ "$RUN_UPDATER" = true ]; then
    echo "$UPDATER_PID" > "$LOG_DIR/updater.pid"
fi

echo "All processes started successfully!"
echo "You can view the HTML page at: $HTML_FILE"
echo "To stop the processes, run: kill \$(cat $LOG_DIR/generator.pid) \$(cat $LOG_DIR/updater.pid)"

# Provide instructions for monitoring
echo ""
echo "To monitor the generation progress:"
echo "  tail -f $LOG_DIR/generation.log"
echo ""
echo "To monitor the HTML updater:"
echo "  tail -f $LOG_DIR/updater.log" 