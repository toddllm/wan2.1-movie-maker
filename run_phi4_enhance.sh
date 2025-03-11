#!/bin/bash

# Script to run Phi-4 prompt enhancement and video generation
# This script enhances prompts using Phi-4 and then generates videos

set -e

# Default values
NUM_PROMPTS=5
VIDEO_DURATION=10
INPUT_PROMPTS="prompts.txt"
ENHANCED_PROMPTS="enhanced_prompts.txt"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -n|--num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    -d|--duration)
      VIDEO_DURATION="$2"
      shift 2
      ;;
    -i|--input)
      INPUT_PROMPTS="$2"
      shift 2
      ;;
    -o|--output)
      ENHANCED_PROMPTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-n|--num-prompts NUM] [-d|--duration SECONDS] [-i|--input INPUT_FILE] [-o|--output OUTPUT_FILE]"
      exit 1
      ;;
  esac
done

# Log file
LOG_FILE="phi4_enhance_generate.log"
echo "Starting Phi-4 enhancement and video generation at $(date)" | tee -a "$LOG_FILE"

# Activate the nexa_venv
if [ -d "$HOME/nexa_venv" ]; then
    echo "Activating nexa_venv virtual environment" | tee -a "$LOG_FILE"
    source "$HOME/nexa_venv/bin/activate"
else
    echo "nexa_venv not found at $HOME/nexa_venv. Please create it first." | tee -a "$LOG_FILE"
    exit 1
fi

# Check if Phi-4 model is set up
if [ ! -d "./phi4_models" ]; then
    echo "Phi-4 model not found. Running setup first..." | tee -a "$LOG_FILE"
    ./run_phi4_setup.sh
    
    # Check if setup was successful
    if [ $? -ne 0 ]; then
        echo "Phi-4 model setup failed. Exiting." | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# Make scripts executable if they're not already
if [ ! -x ./phi4_enhance.py ]; then
    echo "Making phi4_enhance.py executable" | tee -a "$LOG_FILE"
    chmod +x ./phi4_enhance.py
fi

# Check if input prompts file exists
if [ ! -f "$INPUT_PROMPTS" ]; then
    echo "Input prompts file $INPUT_PROMPTS not found. Creating a sample file..." | tee -a "$LOG_FILE"
    cat > "$INPUT_PROMPTS" << EOF
A cat playing with a ball of yarn
A spaceship landing on an alien planet
A medieval castle on a hilltop
A futuristic city with flying cars
A peaceful forest with a stream
EOF
    echo "Created sample prompts file $INPUT_PROMPTS" | tee -a "$LOG_FILE"
fi

# Enhance prompts using Phi-4
echo "Enhancing prompts using Phi-4..." | tee -a "$LOG_FILE"
./phi4_enhance.py batch "$INPUT_PROMPTS" "$ENHANCED_PROMPTS" 2>&1 | tee -a "$LOG_FILE"

# Check if enhancement was successful
if [ ! -f "$ENHANCED_PROMPTS" ]; then
    echo "Enhanced prompts file $ENHANCED_PROMPTS not created. Using original prompts." | tee -a "$LOG_FILE"
    cp "$INPUT_PROMPTS" "$ENHANCED_PROMPTS"
fi

# Select the specified number of prompts
echo "Selecting $NUM_PROMPTS prompts for video generation..." | tee -a "$LOG_FILE"
head -n "$NUM_PROMPTS" "$ENHANCED_PROMPTS" > "selected_prompts.txt"

# Generate videos using the enhanced prompts
echo "Generating videos with duration $VIDEO_DURATION seconds..." | tee -a "$LOG_FILE"
./run_direct_generate.sh "$NUM_PROMPTS" "$VIDEO_DURATION" "selected_prompts.txt" 2>&1 | tee -a "$LOG_FILE"

echo "Video generation process started in the background." | tee -a "$LOG_FILE"
echo "Check direct_generate.log for progress." | tee -a "$LOG_FILE"
echo "Completed at $(date)" | tee -a "$LOG_FILE" 