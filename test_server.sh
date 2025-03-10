#!/bin/bash

# test_server.sh - Script to start Movie Maker on a specified port for testing
# Usage: ./test_server.sh [port]
# Example: ./test_server.sh 5002

# Default port if not specified
PORT=${1:-5002}

echo "Starting Movie Maker test server on port $PORT..."
echo "Access the application at: http://localhost:$PORT"
echo "Press Ctrl+C to stop the server"
echo ""

# Kill any existing process on the specified port
fuser -k $PORT/tcp 2>/dev/null

# Start the application
python app.py --host 0.0.0.0 --port $PORT 