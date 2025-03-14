#!/bin/bash
# Simple script to start a Python HTTP server to listen to the audio files
# NOTE: This is an alternative server and not needed when feedback_server.py is running

cd ~/movie_maker
echo "Starting HTTP server on 0.0.0.0:8080"
echo "NOTE: This server is only needed if feedback_server.py is not running"
echo "To listen to your audio files, open a browser and go to:"
echo "http://localhost:8080/listen.html (from this machine)"
echo "http://YOUR_IP_ADDRESS:8080/listen.html (from other devices on the network)"
echo ""
echo "For the comprehensive voice explorer:"
echo "http://localhost:8080/voice_explorer.html"
echo ""
echo "Your IP address might be one of these:"
ip addr | grep -E "inet .* global" | awk '{print $2}' | cut -d/ -f1
echo ""
echo "Press Ctrl+C to stop the server when done."
python3 -m http.server 8080 --bind 0.0.0.0 