#!/bin/bash

# run_model_setup.sh - Script to download and set up a vision model
# Usage: ./run_model_setup.sh [model_name]

# Default values
MODEL=${1:-"idefics3-8b"}
LOG_FILE="vision_model_setup_$(date +"%Y%m%d_%H%M%S").log"

echo "Starting vision model setup at $(date)"
echo "Model to download: $MODEL"
echo "Log file: $LOG_FILE"

# Create vision_models directory if it doesn't exist
mkdir -p vision_models

# List available models if no model is specified
if [ "$MODEL" == "list" ]; then
    echo "Listing available models..."
    ./setup_vision_model.py --list | tee "$LOG_FILE"
    exit 0
fi

# Download and set up the model
echo "Downloading and setting up $MODEL..."
./setup_vision_model.py --model "$MODEL" 2>&1 | tee "$LOG_FILE"

echo "Vision model setup completed at $(date)"
echo "Check $LOG_FILE for details" 