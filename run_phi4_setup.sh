#!/bin/bash

# Script to set up the Phi-4-multimodal-instruct model
# This script handles the setup of the Phi-4 model for enhancing prompts and image/audio understanding

set -e

# Log file
LOG_FILE="phi4_setup.log"
echo "Starting Phi-4 model setup at $(date)" | tee -a "$LOG_FILE"

# Source HF_TOKEN if available
if [ -f ~/source.me ]; then
    echo "Sourcing token from ~/source.me" | tee -a "$LOG_FILE"
    source ~/source.me
fi

# Check if we have a Hugging Face token
if [ -z "$HF_TOKEN" ]; then
    echo "No HF_TOKEN found in environment. Will attempt to download without token." | tee -a "$LOG_FILE"
    echo "If the model requires authentication, please set HF_TOKEN in your environment or in ~/source.me" | tee -a "$LOG_FILE"
    TOKEN_ARG=""
else
    echo "HF_TOKEN found in environment. Will use for authentication." | tee -a "$LOG_FILE"
    TOKEN_ARG="--token $HF_TOKEN"
fi

# Make the script executable if it's not already
if [ ! -x ./setup_phi4_model.py ]; then
    echo "Making setup_phi4_model.py executable" | tee -a "$LOG_FILE"
    chmod +x ./setup_phi4_model.py
fi

# Run the setup script
echo "Running Phi-4 model setup script" | tee -a "$LOG_FILE"
./setup_phi4_model.py $TOKEN_ARG 2>&1 | tee -a "$LOG_FILE"

# Check if setup was successful
if grep -q "Successfully set up" "$LOG_FILE"; then
    echo "Phi-4 model setup completed successfully!" | tee -a "$LOG_FILE"
    exit 0
else
    echo "Phi-4 model setup may have encountered issues. Check the log file for details." | tee -a "$LOG_FILE"
    exit 1
fi 