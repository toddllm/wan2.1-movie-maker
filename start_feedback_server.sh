#!/bin/bash
# Start the feedback server for CSM Voice Explorer
# This server also serves the voice_explorer.html and listen.html files

cd ~/movie_maker
echo "Starting feedback server on 0.0.0.0:8000"
echo "This server also serves the voice explorer and listen interfaces"
echo ""
echo "To access the voice explorer, open a browser and go to:"
echo "http://localhost:8000/voice_explorer.html (from this machine)"
echo "http://YOUR_IP_ADDRESS:8000/voice_explorer.html (from other devices on the network)"
echo ""
echo "To listen to your audio files, open a browser and go to:"
echo "http://localhost:8000/listen.html (from this machine)"
echo "http://YOUR_IP_ADDRESS:8000/listen.html (from other devices on the network)"
echo ""
echo "Your IP address might be one of these:"
ip addr | grep -E "inet .* global" | awk '{print $2}' | cut -d/ -f1
echo ""
echo "Press Ctrl+C to stop the server when done."
python3 feedback_server.py 