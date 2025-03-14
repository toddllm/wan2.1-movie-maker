#!/bin/bash

# Script to generate 100 more voice samples with 80% female voices and 20% male voices

# Change to the voice_poc directory and activate the virtual environment
cd /home/tdeshane/movie_maker/voice_poc && source venv/bin/activate

# First, generate 80 female voices (80% of total)
echo "Generating 80 female voices..."
python comprehensive_sequential.py \
  --model-id sesame/csm-1b \
  --device cpu \
  --max-samples 80 \
  --random \
  --speakers 2 5 7 \
  --temperatures 1.1 1.3

# Then, generate 20 male voices (20% of total)
echo "Generating 20 male voices..."
python comprehensive_sequential.py \
  --model-id sesame/csm-1b \
  --device cpu \
  --max-samples 20 \
  --random \
  --speakers 0 1 3 4 6 \
  --temperatures 0.9 1.1 1.3

echo "Voice generation complete!"
echo "Generated 100 voice samples (80 female, 20 male)"
echo "To view the samples, start the web server and go to http://localhost:8000/voice_explorer.html" 