#!/bin/bash
# Run Movie Maker Application

# Check if FFmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg is not installed. Please install it first:"
    echo "sudo apt-get update && sudo apt-get install ffmpeg"
    exit 1
fi

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2)
python_major=$(echo $python_version | cut -d'.' -f1)
python_minor=$(echo $python_version | cut -d'.' -f2)

if [ "$python_major" -lt 3 ] || { [ "$python_major" -eq 3 ] && [ "$python_minor" -lt 10 ]; }; then
    echo "Python 3.10 or higher is required. Found version: $python_version"
    exit 1
fi

# Create directories if they don't exist
mkdir -p clips movies

# Check if Wan2.1 model exists
MODEL_DIR=~/development/wan-video/wan2.1/models/Wan2.1-T2V-1.3B
if [ ! -d "$MODEL_DIR" ]; then
    echo "Warning: Wan2.1 model directory not found at $MODEL_DIR"
    echo "Please ensure the model is installed before generating videos."
fi

# Set default values
HOST="0.0.0.0"
PORT="5000"
DEBUG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host=*)
            HOST="${1#*=}"
            shift
            ;;
        --port=*)
            PORT="${1#*=}"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host=HOST] [--port=PORT] [--debug]"
            exit 1
            ;;
    esac
done

# Kill any existing process on the specified port
echo "Checking for existing processes on port $PORT..."
pid=$(lsof -t -i:$PORT)
if [ -n "$pid" ]; then
    echo "Killing existing process (PID: $pid) on port $PORT"
    kill -9 $pid
fi

# Wait a moment for the port to be freed
sleep 1

# Run the application
echo "Starting Movie Maker on http://$HOST:$PORT..."
python3 app.py --host "$HOST" --port "$PORT" $DEBUG 