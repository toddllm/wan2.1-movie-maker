#!/bin/bash
# Run the Movie Maker Beta service on port 5002

# Set the working directory to the script's directory
cd "$(dirname "$0")"

# Check if the app_beta.py file exists
if [ ! -f "app_beta.py" ]; then
    echo "Error: app_beta.py not found!"
    exit 1
fi

# Check if the scoring_system.py file exists
if [ ! -f "scoring_system.py" ]; then
    echo "Error: scoring_system.py not found!"
    exit 1
fi

# Check if the templates directory exists
if [ ! -d "templates" ]; then
    echo "Error: templates directory not found!"
    exit 1
fi

# Check if the required template files exist
if [ ! -f "templates/analysis.html" ] || [ ! -f "templates/preferences.html" ]; then
    echo "Error: Required template files not found!"
    exit 1
fi

# Check if the port is already in use
if netstat -tuln | grep -q ":5002 "; then
    echo "Warning: Port 5002 is already in use. The service may not start properly."
fi

# Run the beta service
echo "Starting Movie Maker Beta service on port 5002..."
python app_beta.py --port 5002 "$@" 