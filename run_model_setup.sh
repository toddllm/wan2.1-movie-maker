#!/bin/bash

# run_model_setup.sh - Script to download and set up a vision model
# Usage: ./run_model_setup.sh [model_name]

# Source the token file if it exists
if [ -f ~/source.me ]; then
    echo "Sourcing token file..."
    source ~/source.me
    echo "HF_TOKEN is set: ${HF_TOKEN:+yes}"
fi

# Default values
MODEL=${1:-"idefics3-8b"}
LOG_FILE="vision_model_setup_$(date +"%Y%m%d_%H%M%S").log"

# List of models to try in order if the specified one fails
MODELS=("idefics3-8b" "cogvlm2" "llava-onevision" "openflamingo")

echo "Starting vision model setup at $(date)"
echo "Initial model to try: $MODEL"
echo "Log file: $LOG_FILE"

# Create vision_models directory if it doesn't exist
mkdir -p vision_models

# List available models if requested
if [ "$MODEL" == "list" ]; then
    echo "Listing available models..."
    ./setup_vision_model.py --list | tee "$LOG_FILE"
    exit 0
fi

# Function to try downloading a model
try_model() {
    local model=$1
    echo "Attempting to download and set up $model..."
    
    if [ -n "$HF_TOKEN" ]; then
        echo "Using Hugging Face token for authentication"
        ./setup_vision_model.py --model "$model" --token "$HF_TOKEN" 2>&1 | tee -a "$LOG_FILE"
    else
        echo "No Hugging Face token found, trying without authentication"
        ./setup_vision_model.py --model "$model" 2>&1 | tee -a "$LOG_FILE"
    fi
    
    # Check if the model was successfully downloaded
    if grep -q "Successfully set up" "$LOG_FILE"; then
        return 0  # Success
    else
        return 1  # Failure
    fi
}

# Try the specified model first
if try_model "$MODEL"; then
    echo "Successfully set up $MODEL"
    echo "Vision model setup completed at $(date)"
    echo "Check $LOG_FILE for details"
    exit 0
fi

echo "Failed to set up $MODEL, trying other models..."

# Try other models in the list
for model in "${MODELS[@]}"; do
    # Skip the model we already tried
    if [ "$model" == "$MODEL" ]; then
        continue
    fi
    
    echo "Trying alternative model: $model"
    if try_model "$model"; then
        echo "Successfully set up $model as a fallback"
        echo "Vision model setup completed at $(date)"
        echo "Check $LOG_FILE for details"
        exit 0
    fi
done

echo "Failed to set up any model. Please check the log file for details."
echo "Vision model setup failed at $(date)"
echo "Check $LOG_FILE for details"
exit 1 