#!/bin/bash

echo "Starting CSM Voice Generation Status Server..."
echo "This will monitor the voice generation progress."
echo "Access the status page at: http://localhost:8000/voice_status.html"
echo ""
echo "Press Ctrl+C to stop the server when done."

# Make sure we're in the movie_maker directory
cd ~/movie_maker

# Make the check_files.py script executable
chmod +x check_files.py

# Start the status server
python3 check_files.py 