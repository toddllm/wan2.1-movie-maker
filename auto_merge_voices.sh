#!/bin/bash

# Script to automatically merge voice samples every few minutes
# This ensures we don't lose previous samples while the voice generation is running

# Set the interval in seconds (5 minutes = 300 seconds)
INTERVAL=300

echo "Starting automatic voice sample merging..."
echo "Will run merge operation every $INTERVAL seconds"
echo "Press Ctrl+C to stop"

while true; do
    echo "$(date): Running merge operation..."
    python3 /home/tdeshane/movie_maker/merge_voice_samples.py --replace
    
    # Count the number of samples
    NUM_SAMPLES=$(grep -c "\"id\":" /home/tdeshane/movie_maker/voice_samples.js)
    echo "$(date): Current number of samples: $NUM_SAMPLES"
    
    echo "Waiting $INTERVAL seconds for next merge..."
    sleep $INTERVAL
done 